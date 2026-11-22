# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grid and single-view prompts for the 24 operating-room appearances used for trocar co-training."""

_SINGLE_VIEW = (
    "A robot-mounted camera observes two three-finger robot hands assembling a surgical trocar over a tray. "
    "The right hand manipulates the sleeve and the left hand manipulates the obturator. Follow the "
    "depth-conditioned camera perspective, instrument alignment and hand positions. "
)

_GRID = (
    "A fixed 2x2 composite shows three synchronized cameras in the same surgical trocar assembly room: head "
    "at top left, left wrist at top right, right wrist at bottom left. Bottom right stays uniformly black. "
    "Preserve the distinct camera poses, panel boundaries, robot, tray and surgical instruments. Restyle "
    "only the background as the same room, equipment, materials and lighting across the views. "
)

_ROOMS = (
    "Two tall vented electronics towers, stacked control units, stainless shelves, and three unmistakable "
    "dark glass displays occupy the background. Large black monitor rectangles alternate with bright "
    "equipment columns and fill most available wall area. Charcoal black, cool white, brushed silver, and "
    "strong cyan-blue accents. Monitor rectangles, ventilation openings, stacked module seams, and shelf "
    "edges. Dim cyan clinical light with bright screen reflections and deep readable equipment shadows.",
    "Tall metal door leaves have strong vertical seams, substantial handles, directional brushed grain, and "
    "a large blue-glass window revealing a brighter adjacent clinical space. The doors and observation "
    "window form a wall-scale vertical landmark rather than an empty wall plane. Brushed silver, cobalt "
    "blue glass, dark graphite seams, and cool white highlights. Vertical door divisions, window framing, "
    "metal grain, and reflected ceiling rectangles. Neutral overhead LEDs create crisp silver reflections "
    "and luminous blue glass.",
    "Glass-front cabinets of varied widths have transparent doors, stainless frames, shelves at different "
    "heights, and loosely grouped blue, teal, and white supply containers behind the glass. Several "
    "transparent cabinets of varied widths sit at offset depths across the background. Aqua glass, teal and "
    "cobalt bins, white packaging, and stainless frames. Varied shelf heights, offset cabinet edges, "
    "loosely grouped supply containers, and soft glass reflections. Bright diffuse clinical light makes the "
    "colored storage and transparent layers clearly readable.",
    "Two large multi-lens lamp heads, blue trim rings, robust white joints, and long curved ceiling arms "
    "dominate the upper room background. Large circles and diagonal boom arcs occupy the ceiling and upper "
    "wall regions across active views. Clean white, pale blue trim, silver joints, and dark concentric lens "
    "centers. Concentric lens rings, repeated LED circles, curved arms, and mechanical joints. Bright "
    "overhead illumination sharply defines lamp hardware without direct glare.",
    "Broad blue drapes cover a long preparation table and a second side surface above reflective cabinet "
    "fronts; sealed white supply packs create sparse high-contrast detail. Large saturated-blue horizontal "
    "bands and bright metal drawer fronts fill the lower background. Deep surgical blue, vivid stainless "
    "silver, cool white, and small violet-blue accents. Fabric folds, long surface edges, drawer seams, "
    "handles, and rectangular supply packs. Neutral-white light preserves rich blue textile texture and "
    "bright metallic reflections.",
    "The large white workstation includes a dominant inactive display, rounded control housings, "
    "translucent breathing circuits, pressure modules, connection fittings, and a stainless base. One bold "
    "dark screen crowns a dense cluster of rounded equipment and visible hose loops. Clinical white, black "
    "glass, translucent gray, teal controls, and polished silver. Circular hose loops, rounded housings, "
    "control clusters, and one large screen rectangle. Soft neutral illumination gives white equipment "
    "dimensional shadows and clean black-screen contrast.",
    "Stainless cupboards and a few broad drawers of varied sizes sit below an offset medical utility rail "
    "with gas sockets, power outlets, gauge faces, and capped connectors in restrained red, yellow, green, "
    "and blue. Asymmetrically grouped cabinetry and one vivid utility strip occupy the available wall "
    "background. Brushed steel, cool white, graphite, and small red, yellow, green, and blue accents. Broad "
    "drawer surfaces, offset handles, separated outlet circles, gauge faces, and irregularly spaced "
    "connectors. Cool side-biased light creates directional metal grain and bright colored outlet details.",
    "A recessed graphite bay contains two tall electronics racks, dark vented panels, cable channels, and "
    "several glowing cyan diagnostic screens without readable text. The dark architectural recess and "
    "luminous display blocks create a strong high-contrast background. Graphite, near-black, electric cyan, "
    "cool white edge light, and dark steel. Vented rack openings, cable channels, luminous rectangles, and "
    "deep alcove edges. Low-key blue illumination with cyan display glow and sharply separated shadow "
    "planes.",
    "A broad blue-tinted glass panel reveals stainless sinks, wall mirrors, soap dispensers, and white "
    "cabinetry in an adjacent illuminated clinical room. The window occupies most wall area and introduces "
    "a clearly layered room beyond the current room. Blue-tinted glass, bright white, stainless silver, "
    "soft aqua, and dark window framing. Window mullions, reflected light panels, sink basins, mirror "
    "edges, and cabinet lines. Bright backlighting through the window contrasts with the moderately exposed "
    "foreground room.",
    "Large white ceiling pendants descend on articulated arms with shelves, sockets, compact dark displays, "
    "cable loops, and color-coded gas connections. Bold vertical columns and diagonal support arms fill "
    "upper and side background regions. Matte white, silver hardware, black displays, teal cable accents, "
    "and muted connector colors. Mechanical joints, stacked shelves, socket clusters, cable loops, and "
    "vertical columns. Balanced cool-white ceiling light produces readable joint shadows and equipment "
    "detail.",
    "Large ceramic wall panels show subtle seams behind paired steel basins, tall sensor faucets, wall "
    "mirrors, and mounted dispenser boxes. The paneled wall and reflective sink zone form a wall-scale "
    "architectural treatment. Light gray ceramic, bright stainless silver, cool white, and subtle aqua "
    "reflections. Subtle wall seams, mirror rectangles, curved faucets, basin ellipses, and dispenser "
    "boxes. Bright task lighting creates crisp tile texture and controlled reflections in the sinks.",
    "Nested carts have tubular legs, caster silhouettes, multiple reflective shelves, and shallow blue "
    "supply trays; dark recessed shelving behind them adds layered depth. Overlapping cart frames and shelf "
    "planes create dense lower-background parallax. Stainless silver, shadowed graphite, cobalt trays, and "
    "cool white highlights. Tubular rectangles, shelf edges, caster circles, nested frames, and stacked "
    "storage planes. Directional cool light produces bright metal edges against a darker storage recess.",
    "A broad storage installation combines one tall white cabinet, several teal-fronted drawers of unequal "
    "sizes, offset stainless handles, and a narrow dark countertop. An asymmetric grouping of doors and "
    "drawers occupies much of the wall with varied spacing. Off-white, saturated teal, dark navy, and "
    "brushed stainless accents. Unequal drawer fronts, one tall door panel, offset handles, and varied "
    "cabinet edges. Even neutral light keeps the teal-white storage pattern vivid and shadow edges clean.",
    "Dark equipment towers contain stacked modules, perforated vents, stainless shelf lips, amber status "
    "windows, and one large powered-off monitor. Dense dark vertical racks and warm indicator blocks create "
    "a background unlike the cool white room variants. Graphite black, warm amber, dark silver, cream-white "
    "walls, and restrained cyan reflections. Vent perforations, module stacks, amber rectangles, shelf "
    "lines, and one black display. Warm-neutral low light emphasizes amber indicators while retaining "
    "clinical realism.",
    "Two large translucent door panels show soft silhouettes of cabinets beyond, framed by cobalt rails, "
    "silver handles, and a visible overhead track. The luminous frosted doors and strong blue frame divide "
    "the wall into unmistakable geometric fields. Frosted white glass, cobalt blue, silver rails, and pale "
    "aqua transmitted light. Large translucent panels, blue frame rectangles, handles, tracks, and diffused "
    "silhouettes. Strong diffuse backlight makes the frosted glass glow while frame edges remain saturated.",
    "A long utility panel carries repeated oxygen and vacuum sockets, pressure gauges, power modules, "
    "coiled blue and green hoses, and compact stainless baskets. The colorful horizontal service band "
    "occupies the central wall instead of leaving a blank surface. Clinical white, blue and green hoses, "
    "yellow outlet caps, graphite gauges, and stainless baskets. Repeating sockets, circular gauges, cable "
    "spirals, wire baskets, and horizontal rails. Bright neutral task light produces crisp colored utility "
    "details and soft wall shadows.",
    "Two large dark monitors, white equipment towers of unequal heights, one stainless door, a curved "
    "storage console, and a few offset ceiling lights create a fully equipped modern room. Two offset "
    "equipment towers and a curved reflective storage console anchor the panoramic background. High-key "
    "white, polished silver, charcoal black, cobalt blue, and pale cyan. Unequal monitor silhouettes, "
    "offset equipment modules, isolated ceiling lights, one door edge, ventilation openings, and curved "
    "cabinet surfaces. Very bright diffuse LEDs preserve details in white equipment and polished metal "
    "without clipping.",
    "Deep blue-gray wall panels surround several cyan-lit displays, narrow ceiling light bars, white "
    "equipment edges, and dark stainless cabinets. Luminous screens and linear ceiling highlights punctuate "
    "broad dark architectural planes. Deep navy, cyan light, charcoal, cool white edges, and dark brushed "
    "steel. Glowing display rectangles, long light strips, panel seams, and cabinet silhouettes. "
    "Underexposed blue ambience with bright cyan screens and narrow controlled highlights.",
    "Warm white composite panels, a long blue-covered side table, satin-finish cabinets, and two dark wall "
    "monitors create a softer but still sterile operating-room background. Cream vertical panels contrast "
    "with saturated blue horizontal surfaces and paired black screens. Warm white, surgical blue, satin "
    "silver, charcoal black, and soft gray. Vertical wall panels, broad fabric folds, monitor rectangles, "
    "and cabinet seams. Warm-neutral LEDs create natural whites, soft fabric shadows, and low-glare metal "
    "reflections.",
    "A large recessed pass-through window has sliding metal panels, offset inner shelves, loosely grouped "
    "blue wrapped packs, and reflective cupboards with drawers of unequal widths below. The dark recessed "
    "hatch and asymmetric lower cabinetry make a strong two-level wall composition. Brushed silver, dark "
    "recess gray, cobalt packaging, and cool white wall trim. Sliding hatch edges, offset shelves, loosely "
    "grouped packages, unequal drawer fronts, and separated handles. Side-biased neutral light emphasizes "
    "the hatch depth and directional stainless texture.",
    "The broad console contains two inactive displays, arrays of physical controls without readable labels, "
    "vent slots, side access panels, and stainless edge trim. One large dark technical block interrupts the "
    "bright wall and occupies most central background pixels. Graphite black, matte white, cyan "
    "reflections, silver trim, and tiny muted control accents. Paired screens, control arrays, vent slots, "
    "panel seams, and strong rectangular framing. Focused cool task lighting separates the black console "
    "from bright modular wall panels.",
    "A few large LED light panels of unequal sizes sit at offset ceiling positions above two circular lamp "
    "heads, articulated white arms, a broad blue drape, and reflective lower cabinets. Strong ceiling "
    "perspective, large circles, and a saturated lower band create three distinct geometric layers. Bright "
    "white, surgical blue, polished silver, pale gray, and dark lens centers. Offset ceiling lights, "
    "concentric lamps, articulated arcs, fabric folds, and broad lower cabinetry. Crisp bright clinical "
    "light creates deep-focus detail across ceiling, equipment, and storage.",
    "Blue-tinted glass panels in slim metal frames reveal layered silhouettes of two mobile monitor carts, "
    "cable loops, equipment shelves, and illuminated ceiling panels in an adjoining bay. Transparent "
    "partition layers and equipment silhouettes create strong depth across the background. Blue-green "
    "glass, charcoal monitors, white carts, silver frames, and bright ceiling reflections. Glass mullions, "
    "reflected rectangles, cart frames, cable curves, and layered silhouettes. Mixed front and back "
    "lighting produces visible glass reflections while preserving the adjoining room detail.",
    "One large and two smaller dark monitors, two equipment towers at different heights, stainless double "
    "doors, one glass-front supply cabinet, a short color-coded utility rail, and curved lower storage form "
    "a rich asymmetric equipment panorama. Distinct equipment groups occupy separated left, center, and "
    "right background zones with irregular gaps of plain wall. Balanced white, charcoal black, brushed "
    "silver, cobalt and teal glass, and small utility colors. Unequal monitor silhouettes, offset equipment "
    "modules, one door opening, sparse glass shelves, isolated utility fittings, and curved cabinet fronts. "
    "Balanced bright clinical illumination keeps every major background group clearly separated and "
    "realistic.",
)

TROCAR_PROMPTS = tuple(_SINGLE_VIEW + room for room in _ROOMS)
TROCAR_GRID_PROMPTS = tuple(_GRID + room for room in _ROOMS)
