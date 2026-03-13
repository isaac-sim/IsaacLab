# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Analyze Python import dependencies across the IsaacLab codebase.

This tool scans Python files to identify import dependencies, detect circular
imports, and generate dependency graphs for visualization and analysis.

Usage:
    # Analyze entire source directory
    python scripts/tools/analyze_dependencies.py --path source/
    
    # Analyze specific module
    python scripts/tools/analyze_dependencies.py --path source/isaaclab/
    
    # Export dependency graph to JSON
    python scripts/tools/analyze_dependencies.py --path source/ --output deps.json
    
    # Check for circular dependencies
    python scripts/tools/analyze_dependencies.py --path source/ --check-circular
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DependencyAnalyzer:
    """Analyzes Python import dependencies in a codebase."""
    
    def __init__(self, root_path: Path):
        """Initialize analyzer with root path.
        
        Args:
            root_path: Root directory to analyze
        """
        self.root_path = root_path.resolve()
        self.dependencies = defaultdict(set)
        self.files_analyzed = 0
        self.imports_found = 0
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Set of imported module names
        """
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"[WARNING] Failed to parse {file_path}: {e}")
        
        return imports
    
    def analyze_directory(self, directory: Path) -> None:
        """Recursively analyze all Python files in directory.
        
        Args:
            directory: Directory to analyze
        """
        for py_file in directory.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            
            relative_path = py_file.relative_to(self.root_path)
            module_name = str(relative_path.with_suffix('')).replace('/', '.')
            
            imports = self.extract_imports(py_file)
            self.dependencies[module_name] = imports
            
            self.files_analyzed += 1
            self.imports_found += len(imports)
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the codebase.
        
        Returns:
            List of circular dependency chains
        """
        def dfs(node: str, visited: Set[str], path: List[str]) -> List[str]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return []
            
            visited.add(node)
            path.append(node)
            
            for dep in self.dependencies.get(node, []):
                if dep in self.dependencies:
                    cycle = dfs(dep, visited, path.copy())
                    if cycle:
                        return cycle
            
            return []
        
        circles = []
        visited = set()
        
        for module in self.dependencies:
            if module not in visited:
                cycle = dfs(module, visited, [])
                if cycle and cycle not in circles:
                    circles.append(cycle)
        
        return circles
    
    def get_dependency_stats(self) -> Dict[str, int]:
        """Calculate dependency statistics.
        
        Returns:
            Dictionary with various statistics
        """
        import_counts = [len(deps) for deps in self.dependencies.values()]
        
        return {
            'total_files': self.files_analyzed,
            'total_modules': len(self.dependencies),
            'total_imports': self.imports_found,
            'avg_imports_per_file': self.imports_found / max(self.files_analyzed, 1),
            'max_imports': max(import_counts) if import_counts else 0,
            'min_imports': min(import_counts) if import_counts else 0,
        }
    
    def get_most_imported(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently imported modules.
        
        Args:
            top_n: Number of top modules to return
            
        Returns:
            List of (module_name, import_count) tuples
        """
        import_freq = defaultdict(int)
        
        for deps in self.dependencies.values():
            for dep in deps:
                import_freq[dep] += 1
        
        return sorted(import_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def export_to_json(self, output_path: Path) -> None:
        """Export dependency graph to JSON.
        
        Args:
            output_path: Path to output JSON file
        """
        export_data = {
            'statistics': self.get_dependency_stats(),
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
            'most_imported': self.get_most_imported(20)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[SUCCESS] Dependency graph exported to: {output_path}")


def print_analysis_report(analyzer: DependencyAnalyzer, check_circular: bool = False):
    """Print formatted analysis report.
    
    Args:
        analyzer: DependencyAnalyzer instance
        check_circular: Whether to check for circular dependencies
    """
    stats = analyzer.get_dependency_stats()
    
    print("\n" + "="*80)
    print("Dependency Analysis Report")
    print("="*80)
    
    print("\n[STATISTICS]")
    print(f"  Files analyzed:        {stats['total_files']}")
    print(f"  Modules found:         {stats['total_modules']}")
    print(f"  Total imports:         {stats['total_imports']}")
    print(f"  Avg imports per file:  {stats['avg_imports_per_file']:.2f}")
    print(f"  Max imports (file):    {stats['max_imports']}")
    print(f"  Min imports (file):    {stats['min_imports']}")
    
    print("\n[MOST IMPORTED MODULES]")
    print("-"*80)
    most_imported = analyzer.get_most_imported(15)
    for i, (module, count) in enumerate(most_imported, 1):
        print(f"  {i:2}. {module:30} ({count} imports)")
    
    if check_circular:
        print("\n[CIRCULAR DEPENDENCY CHECK]")
        print("-"*80)
        circles = analyzer.find_circular_dependencies()
        
        if circles:
            print(f"  [WARNING] Found {len(circles)} circular dependency chain(s):")
            for i, circle in enumerate(circles, 1):
                print(f"\n  Chain {i}:")
                for j, module in enumerate(circle):
                    if j < len(circle) - 1:
                        print(f"    {module} ->")
                    else:
                        print(f"    {module}")
        else:
            print("  [OK] No circular dependencies detected")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze Python import dependencies.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)"
    )
    parser.add_argument(
        "--check-circular",
        action="store_true",
        help="Check for circular dependencies"
    )
    
    args = parser.parse_args()
    
    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f"[ERROR] Path does not exist: {path}")
        sys.exit(1)
    
    if not path.is_dir():
        print(f"[ERROR] Path is not a directory: {path}")
        sys.exit(1)
    
    # Analyze dependencies
    print(f"Analyzing dependencies in: {path}")
    analyzer = DependencyAnalyzer(path)
    analyzer.analyze_directory(path)
    
    # Print report
    print_analysis_report(analyzer, args.check_circular)
    
    # Export if requested
    if args.output:
        analyzer.export_to_json(Path(args.output))


if __name__ == "__main__":
    main()
