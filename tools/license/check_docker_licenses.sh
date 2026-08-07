#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 CURRENT_REPORT BASE_REPORT BUNDLED_COMPONENTS SUMMARY_OUTPUT SUMMARY_TITLE BASE_IMAGE_LABEL" >&2
    exit 2
fi

CURRENT_REPORT="$1"
BASE_REPORT="$2"
BUNDLED_COMPONENTS="$3"
SUMMARY_OUTPUT="$4"
SUMMARY_TITLE="$5"
BASE_IMAGE_LABEL="$6"
EXCEPTIONS_FILE=.github/workflows/license-exceptions.json
FINDINGS_DELIMITER=$'\x1f'
ALLOWED_LICENSES=(
    "0bsd"
    "apache-2.0"
    "apache-2.0 and cnri-python"
    "apache-2.0 with llvm-exception"
    "bsd-0-clause"
    "bsd-1-clause"
    "bsd-2-clause"
    "bsd-2-clause and apache-2.0 with llvm-exception"
    "bsd-2-clause-netbsd"
    "bsd-3-clause"
    "bsd-3-clause and 0bsd and mit and zlib and cc0-1.0"
    "bsd-4-clause"
    "bsd-4-clause-uc"
    "cc0-1.0"
    "isc"
    "mit"
    "mit or apache-2.0"
    "mit/x11"
    "x11"
    "zlib"
)

for report in "$CURRENT_REPORT" "$BASE_REPORT"; do
    if ! jq empty "$report"; then
        echo "::error::Failed to parse Trivy report: $report"
        exit 1
    fi
    if ! jq -e 'any(.Results[]; .Class == "license")' "$report" >/dev/null; then
        echo "::error::Trivy report contains no license results: $report"
        exit 1
    fi
done

# Refuse to pass if Trivy did not inventory both dependency types in the final
# image. Loose-file licenses are intentionally excluded.
for package_class in os-pkgs lang-pkgs; do
    if ! jq -e --arg class "$package_class" \
        'any(.Results[]; .Class == $class)' "$CURRENT_REPORT" >/dev/null; then
        echo "::error::Trivy report is missing the $package_class inventory"
        exit 1
    fi
done

extract_findings() {
    jq -r '
      (
        [
          .Results[]
          | select(.Class == "os-pkgs" or .Class == "lang-pkgs")
          | (
              if .Class == "os-pkgs"
              then "OS Packages"
              else .Target
              end
            ) as $package_type
          | .Packages[]?
          | {
              key: (
                ($package_type + "\u0000" + .Name)
                | ascii_downcase
              ),
              value: (.Version // "")
            }
        ]
        | from_entries
      ) as $versions
      |
      .Results[]
      | select(.Class == "license")
      | .Target as $package_type
      | .Licenses[]?
      | select((.PkgName // "") != "")
      | [
          $package_type,
          .PkgName,
          (
            $versions[
              (($package_type + "\u0000" + .PkgName) | ascii_downcase)
            ] // ""
          ),
          .Name,
          .Category,
          .Severity,
          ""
        ]
      | map(tostring)
      | join("\u001f")
    ' "$1" | sort -u
}

extract_findings "$CURRENT_REPORT" > current-findings.tsv
if [[ ! -s current-findings.tsv ]]; then
    echo "::error::Trivy found no package licenses in the final image"
    exit 1
fi
extract_findings "$BASE_REPORT" > base-findings.tsv
if [[ ! -s base-findings.tsv ]]; then
    echo "::error::Trivy found no package licenses in the $BASE_IMAGE_LABEL"
    exit 1
fi

tr '\t' '\037' < "$BUNDLED_COMPONENTS" >> current-findings.tsv
sort -u -o current-findings.tsv current-findings.tsv
cut -d "$FINDINGS_DELIMITER" -f1-4 base-findings.tsv \
    | tr '[:upper:]' '[:lower:]' > base-identities.tsv

TOTAL_FINDINGS=$(wc -l < current-findings.tsv)
INHERITED_FINDINGS=0
ALLOWED_FINDINGS=0
NVIDIA_FINDINGS=0
EXCEPTED_FINDINGS=0
FAILED_FINDINGS=0
: > license-violations.md
: > reviewed-license-declarations.md

record_violation() {
    local reason="$1"
    FAILED_FINDINGS=$((FAILED_FINDINGS + 1))
    printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
        "$package_type" "$package" "$version" "$license" "$category" \
        "$severity" "$reason" >> license-violations.md
}

while IFS="$FINDINGS_DELIMITER" read -r \
    package_type package version license category severity detected_linkage; do
    if [[ -z "$version" ]]; then
        record_violation "Package is missing from Trivy's version inventory"
        continue
    fi

    identity=$(printf '%s\x1f%s\x1f%s\x1f%s' \
        "$package_type" "$package" "$version" "$license" \
        | tr '[:upper:]' '[:lower:]')

    if grep -Fqx -- "$identity" base-identities.tsv; then
        INHERITED_FINDINGS=$((INHERITED_FINDINGS + 1))
        continue
    fi

    if [[ "${package,,}" == nvidia* ]]; then
        NVIDIA_FINDINGS=$((NVIDIA_FINDINGS + 1))
        continue
    fi

    license_is_allowed=false
    for allowed_license in "${ALLOWED_LICENSES[@]}"; do
        if [[ "${license,,}" == "$allowed_license" ]]; then
            license_is_allowed=true
            break
        fi
    done
    if "$license_is_allowed"; then
        ALLOWED_FINDINGS=$((ALLOWED_FINDINGS + 1))
        continue
    fi

    exception="$(
        jq -c \
            --arg package "$package" \
            --arg license "$license" \
            --arg package_type "$package_type" '
          first(
            .[]
            | select(
                (.package | ascii_downcase) == ($package | ascii_downcase)
                and (
                  (.package_type == $package_type)
                  or ($package_type == "Python" and .package_type == null)
                )
                and (
                  (.license == null)
                  or ((.license | ascii_downcase) == ($license | ascii_downcase))
                  or any(.license_aliases[]?;
                    ascii_downcase == ($license | ascii_downcase)
                  )
                )
            )
          )
        ' "$EXCEPTIONS_FILE"
    )"
    if [[ -z "$exception" ]]; then
        record_violation "No reviewed exception"
        continue
    fi

    usage="$(jq -r '.usage // ""' <<< "$exception")"
    interaction="$(jq -r '.interaction // ""' <<< "$exception")"
    declared_linkage="$(jq -r '.linkage // ""' <<< "$exception")"
    selected_license="$(jq -r '.selected_license // ""' <<< "$exception")"
    effective_license="${selected_license:-$license}"
    linkage="${detected_linkage:-$declared_linkage}"

    if [[ -z "$usage" ]]; then
        record_violation "Reviewed exception is missing usage"
        continue
    fi
    case "$interaction" in
        standalone_process | standalone_process_dependency | not_loaded | build_input | \
            same_process_dynamic | same_process_static)
            ;;
        *)
            record_violation "Reviewed exception has missing or invalid interaction"
            continue
            ;;
    esac

    normalized_license="${effective_license,,}"
    is_lgpl=false
    is_gpl=false
    if [[ "$normalized_license" == *"lgpl"* ||
        "$normalized_license" == *"lesser general public license"* ]]; then
        is_lgpl=true
    elif [[ "$normalized_license" == *"gpl"* ||
        "$normalized_license" == *"general public license"* ]]; then
        is_gpl=true
    fi

    if "$is_gpl" &&
        [[ "$interaction" == same_process_dynamic ||
            "$interaction" == same_process_static ]]; then
        record_violation "GPL component is declared in the Isaac Lab process"
        continue
    fi

    if "$is_lgpl" && [[ "$linkage" == "static" ]]; then
        relinking_materials="$(jq -r '.relinking_materials // ""' <<< "$exception")"
        relinking_instructions="$(jq -r '.relinking_instructions // ""' <<< "$exception")"
        if [[ -z "$relinking_materials" || -z "$relinking_instructions" ]]; then
            record_violation "Statically linked LGPL component lacks relinking materials"
            continue
        fi
    fi

    EXCEPTED_FINDINGS=$((EXCEPTED_FINDINGS + 1))
    printf '| %s | %s | %s | %s | %s | %s |\n' \
        "$package_type" "$package" "$version" "$effective_license" \
        "$usage" "$interaction" >> reviewed-license-declarations.md
done < current-findings.tsv

{
    echo "## $SUMMARY_TITLE"
    echo
    if [[ "$FAILED_FINDINGS" -eq 0 ]]; then
        echo "**Passed.** No unapproved dependency licenses were introduced by the final image."
    else
        echo "**Failed.** Found $FAILED_FINDINGS unapproved dependency license finding(s)."
    fi
    echo
    echo "| Result | Count |"
    echo "|---|---:|"
    echo "| Total package-license findings | $TOTAL_FINDINGS |"
    echo "| Inherited at the same version from $BASE_IMAGE_LABEL | $INHERITED_FINDINGS |"
    echo "| Allowed permissive licenses | $ALLOWED_FINDINGS |"
    echo "| NVIDIA package policy | $NVIDIA_FINDINGS |"
    echo "| Reviewed exceptions | $EXCEPTED_FINDINGS |"
    echo "| Violations | $FAILED_FINDINGS |"
    if [[ -s reviewed-license-declarations.md ]]; then
        echo
        echo "### Reviewed dependency declarations"
        echo
        echo "| Package type | Package | Version | Effective license | Usage | Interaction |"
        echo "|---|---|---|---|---|---|"
        sort -u reviewed-license-declarations.md
    fi
    if [[ "$FAILED_FINDINGS" -gt 0 ]]; then
        echo
        echo "### Unapproved licenses"
        echo
        echo "| Package type | Package | Version | License | Category | Severity | Reason |"
        echo "|---|---|---|---|---|---|---|"
        cat license-violations.md
    fi
} | tee "$SUMMARY_OUTPUT"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$SUMMARY_OUTPUT" >> "$GITHUB_STEP_SUMMARY"
fi

if [[ "$FAILED_FINDINGS" -gt 0 ]]; then
    exit 1
fi
