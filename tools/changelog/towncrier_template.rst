{# Emits fragment bodies verbatim under ``^``-underlined type headings and hides the skip and
   tier types. The final expression adds the second blank line kept before each older version. #}

{% for section, _ in sections.items() %}
{% for category, val in definitions.items() if category in sections[section] and definitions[category]['showcontent'] %}
{{ definitions[category]['name'] }}
{{ underlines[0] * definitions[category]['name']|length }}

{% for text, values in sections[section][category].items() %}
{{ text }}
{% endfor %}

{% endfor %}
{% endfor %}
{{ "\n" }}
