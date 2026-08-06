"""Build MuJoCo-ready meshes and XML from the raw SolidWorks STL export."""

import json
import math
import os
import re
import struct
from collections import defaultdict


ROOT = os.path.dirname(os.path.abspath(__file__))
MESH_SRC_DIR = os.path.join(ROOT, "meshes")
MESH_OUT_DIR = os.path.join(ROOT, "meshes_mj")
XML_OUT = os.path.join(ROOT, "wheeled_infantry.xml")


ROOT_ORIGIN_MM = (1134.001, 423.636, 1060.164)

# Pivot axes extracted from the cylindrical faces of the SolidWorks STL export.
LEFT_HIP_MM = (1311.501, 423.660, 1060.150)
RIGHT_HIP_MM = (956.501, 423.660, 1060.150)
LEFT_ACTIVE_ROOT_MM = (1315.751, 423.660, 1060.150)
RIGHT_ACTIVE_ROOT_MM = (952.251, 423.660, 1060.150)
LEFT_ACTIVE_LONG_MM = (1327.501, 359.342, 990.916)
RIGHT_ACTIVE_LONG_MM = (940.501, 359.342, 990.916)
LEFT_LONG_NODE_MM = (1315.251, 270.660, 1060.147)
RIGHT_LONG_NODE_MM = (952.751, 270.660, 1060.147)
LEFT_NODE_UPPER_MM = (1326.501, 359.335, 1129.378)
RIGHT_NODE_UPPER_MM = (941.501, 359.335, 1129.378)
LEFT_NODE_PARALLEL_MM = (1326.501, 419.180, 1154.752)
RIGHT_NODE_PARALLEL_MM = (941.501, 419.180, 1154.752)
LEFT_KNEE_MM = (1329.001, 280.720, 1213.990)
RIGHT_KNEE_MM = (939.001, 280.720, 1213.990)
LEFT_KNEE_PARALLEL_MM = (1327.501, 340.560, 1239.370)
RIGHT_KNEE_PARALLEL_MM = (940.501, 340.560, 1239.370)
LEFT_PARALLEL_NODE_MM = (1315.251, 419.180, 1154.752)
RIGHT_PARALLEL_NODE_MM = (952.751, 419.180, 1154.752)
LEFT_WHEEL_MM = (1354.490, 83.706, 1060.173)
RIGHT_WHEEL_MM = (913.490, 83.643, 1060.195)
LEFT_UPPER_IDLER_MM = (1300.279, 423.698, 1060.132)
RIGHT_UPPER_IDLER_MM = (967.723, 423.698, 1060.132)
LEFT_ACTIVE_IDLER_MM = (1291.273, 423.672, 1060.143)
RIGHT_ACTIVE_IDLER_MM = (976.729, 423.672, 1060.138)

LEFT_NODE_UPPER_CONNECT_MM = (1320.501, 359.335, 1129.378)
RIGHT_NODE_UPPER_CONNECT_MM = (947.501, 359.335, 1129.378)
LEFT_NODE_PARALLEL_CONNECT_MM = (1320.876, 419.180, 1154.752)
RIGHT_NODE_PARALLEL_CONNECT_MM = (947.126, 419.180, 1154.752)


DENSITIES = {
    "aluminum": 2700.0,
    "steel": 7850.0,
    "motor": 4300.0,
    "plastic": 1240.0,
    "carbon": 1650.0,
    "electronics": 2100.0,
    "battery": 2300.0,
}
DYNAMIC_MASS_SCALE = 0.22


BODY_ORIGINS_MM = {
    "base": ROOT_ORIGIN_MM,
    "l_upper": LEFT_HIP_MM,
    "l_active": LEFT_ACTIVE_ROOT_MM,
    "l_upper_idler": LEFT_UPPER_IDLER_MM,
    "l_active_idler": LEFT_ACTIVE_IDLER_MM,
    "r_upper": RIGHT_HIP_MM,
    "r_active": RIGHT_ACTIVE_ROOT_MM,
    "r_upper_idler": RIGHT_UPPER_IDLER_MM,
    "r_active_idler": RIGHT_ACTIVE_IDLER_MM,
    "l_lower": LEFT_KNEE_MM,
    "l_long": LEFT_ACTIVE_LONG_MM,
    "l_parallel": LEFT_KNEE_PARALLEL_MM,
    "l_node": LEFT_NODE_PARALLEL_MM,
    "r_lower": RIGHT_KNEE_MM,
    "r_long": RIGHT_ACTIVE_LONG_MM,
    "r_parallel": RIGHT_KNEE_PARALLEL_MM,
    "r_node": RIGHT_NODE_PARALLEL_MM,
    "l_wheel": LEFT_WHEEL_MM,
    "r_wheel": RIGHT_WHEEL_MM,
    "l_wheel_assembly": LEFT_WHEEL_MM,
    "r_wheel_assembly": RIGHT_WHEEL_MM,
}


BODY_DISPLAY_NAMES = {
    "base": "base",
    "l_upper": "left_upper_leg",
    "l_upper_idler": "left_upper_idler",
    "l_active_idler": "left_active_idler",
    "r_upper": "right_upper_leg",
    "r_upper_idler": "right_upper_idler",
    "r_active_idler": "right_active_idler",
    "l_lower": "left_lower_leg",
    "r_lower": "right_lower_leg",
    "l_wheel": "left_wheel",
    "r_wheel": "right_wheel",
}


SIDE_SPLIT_X = ROOT_ORIGIN_MM[0]
MAX_FACES_PER_MESH = 200000
SINGLE_MESH_OUTPUTS = {
    "l_wheel_assembly": ("left_wheel_complete", "left_wheel_complete.stl"),
    "r_wheel_assembly": ("right_wheel_complete", "right_wheel_complete.stl"),
}
SPECIAL_BODY_CENTERS_MM = (
    ("l_upper_idler", LEFT_UPPER_IDLER_MM),
    ("r_upper_idler", RIGHT_UPPER_IDLER_MM),
    ("l_active_idler", LEFT_ACTIVE_IDLER_MM),
    ("r_active_idler", RIGHT_ACTIVE_IDLER_MM),
)

MERGED_BODY_TARGETS = {
    "l_upper_idler": "l_upper",
    "l_active_idler": "l_upper",
    "r_upper_idler": "r_upper",
    "r_active_idler": "r_upper",
}


def merged_body_key(body):
    return MERGED_BODY_TARGETS.get(body, body)


def wheel_assembly_body(row, body, material):
    if body == "l_wheel":
        return "l_wheel_assembly"
    if body == "r_wheel":
        return "r_wheel_assembly"

    center = row["center_mm"]
    bbox = row["bbox_mm"]
    is_axle_holder = (
        material == "motor"
        and abs(center[1] - LEFT_WHEEL_MM[1]) < 4.0
        and abs(center[2] - LEFT_WHEEL_MM[2]) < 4.0
        and 14.0 <= bbox[0] <= 16.0
        and 50.0 <= bbox[1] <= 60.0
        and 50.0 <= bbox[2] <= 60.0
    )
    if is_axle_holder:
        return "l_wheel_assembly" if center[0] > SIDE_SPLIT_X else "r_wheel_assembly"
    return merged_body_key(body)


def read_binary_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        tris = []
        for _ in range(count):
            vals = struct.unpack("<12fH", f.read(50))
            tris.append((vals[3:6], vals[6:9], vals[9:12]))
        return header, tris


def write_binary_stl(path, triangles):
    with open(path, "wb") as f:
        header = b"generated for mujoco".ljust(80, b"\0")
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            normal = compute_normal(a, b, c)
            f.write(struct.pack("<12fH", *(normal + a + b + c + (0,))))


def special_body_for_center(center_mm, tol_mm=0.2):
    for body_name, target in SPECIAL_BODY_CENTERS_MM:
        if all(abs(center_mm[i] - target[i]) <= tol_mm for i in range(3)):
            return body_name
    return None


def compute_normal(a, b, c):
    ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (nx / n, ny / n, nz / n)


def mesh_stats(path):
    with open(path, "rb") as f:
        f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        volume = 0.0
        cx = cy = cz = 0.0
        minp = [1e100, 1e100, 1e100]
        maxp = [-1e100, -1e100, -1e100]
        total_pts = 0
        for _ in range(count):
            vals = struct.unpack("<12fH", f.read(50))
            a = vals[3:6]
            b = vals[6:9]
            c = vals[9:12]
            volume += signed_tet_volume(a, b, c)
            for p in (a, b, c):
                for i, v in enumerate(p):
                    if v < minp[i]:
                        minp[i] = v
                    if v > maxp[i]:
                        maxp[i] = v
                cx += p[0]
                cy += p[1]
                cz += p[2]
                total_pts += 1
    center = (cx / total_pts, cy / total_pts, cz / total_pts)
    bbox = tuple(maxp[i] - minp[i] for i in range(3))
    return abs(volume), center, bbox


def signed_tet_volume(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    ) / 6.0


def contains_any(text, keywords):
    return any(k in text for k in keywords)


def is_core_dynamic_part(name):
    core_keywords = [
        " 澶ц吙-",
        "Mirror澶ц吙-",
        " 灏忚吙-",
        "Mirror灏忚吙-",
        "灏忚吙涓诲姩杩炴潌",
        "鑶濆叧鑺傞暱杩炴潌",
        "knee_parallel_link",
        "knee_node_link",
        "120mm_arc_wheel",
    ]
    return contains_any(name, core_keywords)


def material_for(name):
    if contains_any(name, ["榛戣壊鐢垫睜", "鐢垫睜"]):
        return "battery"
    if contains_any(name, ["8009", "RAU_5005", "DM_8116", "3508", "3519", "鏃犲埛", "鍑忛€熺", "婊戠幆MT2042"]):
        return "motor"
    if contains_any(name, ["bearing", "bolt", "nut", "washer", "shaft"]):
        return "steel"
    if contains_any(name, ["纰崇", "鐜荤氦"]):
        return "carbon"
    if contains_any(name, ["PCB", "VT03", "ZX360", "camera", "capacitor"]):
        return "electronics"
    if contains_any(name, ["printed", "shell", "spacer", "stop", "protect", "fill"]):
        return "plastic"
    return "aluminum"


def wheel_family(name):
    if "NW P15.8杞吙鍑忛€熺-1 " in name or "120mm寮у舰杞?1" in name:
        return "l_wheel"
    if "NW P15.8杞吙鍑忛€熺(2)-1" in name or "120mm寮у舰杞?2" in name:
        return "r_wheel"
    return None


def side_from_name(name, center_x):
    if "Mirror" in name:
        return "r"
    if center_x > SIDE_SPLIT_X + 10.0:
        return "l"
    if center_x < SIDE_SPLIT_X - 10.0:
        return "r"
    return None


def body_for(name, center_x, center_y):
    if "楂嬪叧鑺?3" not in name:
        return "base"

    side = side_from_name(name, center_x)
    if not side:
        return "base"

    if not is_core_dynamic_part(name):
        return "base"

    wheel_group = wheel_family(name)
    if wheel_group:
        return wheel_group

    if "婢堆嗗悪娴犲骸濮╅柧鎹愮枂" in name:
        return f"{side}_upper_idler"
    if "active_idler" in name:
        return f"{side}_active_idler"

    upper_keys = [
        "澶ц吙-",
        "澶ц吙浠庡姩閾捐疆",
        "灏忚吙涓诲姩杩炴潌",
        "婊戠幆MT2042",
        "lower_sprocket",
    ]
    lower_keys = [
        "灏忚吙-",
        "灏忚吙浠庡姩閾捐疆",
        "knee_joint",
        "纰崇",
        "armor_tube",
        "娉曞叞杞存壙F698",
        "鑵块暱闄愪綅",
    ]
    base_keys = [
        "8009",
        "RAU_5005",
        "DM_8116",
        "hip_main_shaft",
        "楂嬪叧鑺傜數鏈哄浐瀹氭澘",
        "楂嬪叧鑺傜缁勮酱鎵垮骇",
        "楂嬪叧鑺傝酱涓婂帇绱х洊",
        "楂嬪叧鑺傝酱涓嬪帇绱х洊",
        "楂嬪叧鑺備富杞村帇绱ц灪姣嶇洊",
        "high_precision_bearing",
        "35-44-5.5纰楃粍杞存壙",
        "42-52-7纰楃粍杞存壙",
        "纰楃粍杞存壙32.7-41.8-6",
        "闃叉澗铻烘瘝",
        "M5-",
        "M5.",
    ]

    if contains_any(name, base_keys):
        return "base"
    if "灏忚吙涓诲姩杩炴潌" in name:
        return f"{side}_active"
    if "鑶濆叧鑺傞暱杩炴潌" in name:
        return f"{side}_long"
    if "knee_parallel_link" in name:
        return f"{side}_parallel"
    if "knee_node_link" in name:
        return f"{side}_node"
    if contains_any(name, upper_keys):
        return f"{side}_upper"
    if contains_any(name, lower_keys):
        return f"{side}_lower"
    if center_y < 330.0:
        return f"{side}_lower"
    return "base"


def to_body_key(prefixed):
    if prefixed == "l_upper":
        return "l_upper"
    if prefixed == "l_upper_idler":
        return "l_upper_idler"
    if prefixed == "l_active":
        return "l_active"
    if prefixed == "l_active_idler":
        return "l_active_idler"
    if prefixed == "r_upper":
        return "r_upper"
    if prefixed == "r_upper_idler":
        return "r_upper_idler"
    if prefixed == "r_active":
        return "r_active"
    if prefixed == "r_active_idler":
        return "r_active_idler"
    if prefixed == "l_lower":
        return "l_lower"
    if prefixed == "l_long":
        return "l_long"
    if prefixed == "l_parallel":
        return "l_parallel"
    if prefixed == "l_node":
        return "l_node"
    if prefixed == "r_lower":
        return "r_lower"
    if prefixed == "r_long":
        return "r_long"
    if prefixed == "r_parallel":
        return "r_parallel"
    if prefixed == "r_node":
        return "r_node"
    if prefixed == "l_wheel":
        return "l_wheel"
    if prefixed == "r_wheel":
        return "r_wheel"
    return "base"


def rel_vec_mm(a, b):
    return tuple((a[i] - b[i]) * 0.001 for i in range(3))


def fmt3(v):
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def chunk_triangles(triangles, chunk_size=MAX_FACES_PER_MESH):
    for i in range(0, len(triangles), chunk_size):
        yield triangles[i : i + chunk_size]


def reduce_triangles(triangles, max_faces=MAX_FACES_PER_MESH):
    if len(triangles) <= max_faces:
        return triangles
    stride = len(triangles) / max_faces
    return [triangles[int(index * stride)] for index in range(max_faces)]


def should_export_group(body, material):
    return True


def should_include_row(row, body):
    return True


def reset_output_dir():
    os.makedirs(MESH_OUT_DIR, exist_ok=True)
    for fname in os.listdir(MESH_OUT_DIR):
        if fname.lower().endswith(".stl"):
            os.remove(os.path.join(MESH_OUT_DIR, fname))


def export_grouped_meshes(grouped):
    asset_entries = []
    body_geom_map = defaultdict(list)
    single_body_items = defaultdict(list)

    for (body, material), items in sorted(grouped.items()):
        if not should_export_group(body, material):
            continue
        if body in SINGLE_MESH_OUTPUTS:
            single_body_items[body].extend(items)
            continue
        origin = BODY_ORIGINS_MM[body]
        triangles = []
        for item in items:
            _, tris = read_binary_stl(item["path"])
            for a, b, c in tris:
                triangles.append(
                    (
                        (a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]),
                        (b[0] - origin[0], b[1] - origin[1], b[2] - origin[2]),
                        (c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]),
                    )
                )

        chunks = list(chunk_triangles(triangles))
        for idx, chunk in enumerate(chunks, start=1):
            suffix = f"_part{idx}" if len(chunks) > 1 else ""
            mesh_name = f"{body}_{material}{suffix}"
            mesh_file = f"{mesh_name}.stl"
            write_binary_stl(os.path.join(MESH_OUT_DIR, mesh_file), chunk)
            asset_entries.append((mesh_name, mesh_file))
            body_geom_map[body].append((mesh_name, material))

    for body, items in sorted(single_body_items.items()):
        origin = BODY_ORIGINS_MM[body]
        triangles = []
        for item in items:
            _, tris = read_binary_stl(item["path"])
            for a, b, c in tris:
                triangles.append(
                    (
                        (a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]),
                        (b[0] - origin[0], b[1] - origin[1], b[2] - origin[2]),
                        (c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]),
                    )
                )
        mesh_name, mesh_file = SINGLE_MESH_OUTPUTS[body]
        triangles = reduce_triangles(triangles)
        write_binary_stl(os.path.join(MESH_OUT_DIR, mesh_file), triangles)
        asset_entries.append((mesh_name, mesh_file))
        body_geom_map[body].append((mesh_name, "aluminum"))

    return asset_entries, body_geom_map


def collect_meshes():
    rows = []
    for fname in sorted(os.listdir(MESH_SRC_DIR)):
        if not fname.lower().endswith(".stl"):
            continue
        path = os.path.join(MESH_SRC_DIR, fname)
        volume_mm3, center_mm, bbox_mm = mesh_stats(path)
        rows.append(
            {
                "name": fname,
                "path": path,
                "volume_mm3": volume_mm3,
                "center_mm": center_mm,
                "bbox_mm": bbox_mm,
            }
        )
    return rows


def generate_outputs():
    reset_output_dir()
    rows = collect_meshes()

    grouped = defaultdict(list)
    body_mass = defaultdict(float)
    material_mass = defaultdict(float)

    for row in rows:
        material = material_for(row["name"])
        body = special_body_for_center(row["center_mm"])
        if body is None:
            body = to_body_key(body_for(row["name"], row["center_mm"][0], row["center_mm"][1]))
        body = wheel_assembly_body(row, body, material)
        density = DENSITIES[material] * DYNAMIC_MASS_SCALE
        mass = density * (row["volume_mm3"] * 1e-9)

        row["material"] = material
        row["body"] = body
        row["mass_est"] = mass

        if not should_include_row(row, body):
            continue

        grouped[(body, material)].append(row)
        body_mass[body] += mass
        material_mass[material] += mass

    summary = {
        "body_mass_kg": dict(sorted(body_mass.items())),
        "material_mass_kg": dict(sorted(material_mass.items())),
    }
    with open(os.path.join(ROOT, "classification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    asset_entries, body_geom_map = export_grouped_meshes(grouped)

    xml = build_xml(body_geom_map)
    with open(XML_OUT, "w", encoding="utf-8") as f:
        f.write(xml)


def build_body_geom_xml(body_key, indent="      "):
    out = []
    is_wheel_visual = body_key in {"l_wheel_assembly", "r_wheel_assembly"}
    # Meshes are visual/mass geometry only.  Wheel and linkage proxy geoms
    # below define the complete collision model, preventing decorative CAD
    # meshes from catching the ground during a jump or landing.
    contype = "0"
    conaffinity = "0"
    for mesh_name, material in sorted(body_geom_map_global[body_key]):
        density = DENSITIES[material]
        mass_attr = 'mass="0"' if body_key == "base" else f'density="{density:.1f}"'
        rgba = {
            "aluminum": "0.74 0.76 0.79 1",
            "steel": "0.45 0.47 0.50 1",
            "motor": "0.27 0.29 0.31 1",
            "plastic": "0.16 0.16 0.18 1",
            "carbon": "0.08 0.08 0.09 1",
            "electronics": "0.12 0.38 0.16 1",
            "battery": "0.18 0.18 0.18 1",
        }[material]
        out.append(
            f'{indent}<geom type="mesh" mesh="{mesh_name}" {mass_attr} rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}" group="0"/>'
        )
    return "\n".join(out)


def build_xml(body_geom_map):
    global body_geom_map_global
    body_geom_map_global = body_geom_map

    asset_lines = []
    for mesh_name, mesh_file in sorted(asset_entries_global):
        asset_lines.append(
            f'    <mesh name="{mesh_name}" file="{mesh_file}" scale="0.001 0.001 0.001"/>'
        )

    l_hip = rel_vec_mm(LEFT_HIP_MM, ROOT_ORIGIN_MM)
    r_hip = rel_vec_mm(RIGHT_HIP_MM, ROOT_ORIGIN_MM)
    l_upper_idler = rel_vec_mm(LEFT_UPPER_IDLER_MM, LEFT_HIP_MM)
    r_upper_idler = rel_vec_mm(RIGHT_UPPER_IDLER_MM, RIGHT_HIP_MM)
    l_active_idler = rel_vec_mm(LEFT_ACTIVE_IDLER_MM, LEFT_ACTIVE_ROOT_MM)
    r_active_idler = rel_vec_mm(RIGHT_ACTIVE_IDLER_MM, RIGHT_ACTIVE_ROOT_MM)
    l_active_from_upper = rel_vec_mm(LEFT_ACTIVE_ROOT_MM, LEFT_HIP_MM)
    r_active_from_upper = rel_vec_mm(RIGHT_ACTIVE_ROOT_MM, RIGHT_HIP_MM)
    l_long_from_active = rel_vec_mm(LEFT_ACTIVE_LONG_MM, LEFT_ACTIVE_ROOT_MM)
    r_long_from_active = rel_vec_mm(RIGHT_ACTIVE_LONG_MM, RIGHT_ACTIVE_ROOT_MM)
    l_node_from_parallel = rel_vec_mm(LEFT_NODE_PARALLEL_MM, LEFT_KNEE_PARALLEL_MM)
    r_node_from_parallel = rel_vec_mm(RIGHT_NODE_PARALLEL_MM, RIGHT_KNEE_PARALLEL_MM)
    # The lower link is mounted to the chassis at its own pivot, like the rear
    # branch in ok.xml.  It must not be a child of the upper-link body.
    l_lower_from_root = rel_vec_mm(LEFT_KNEE_MM, ROOT_ORIGIN_MM)
    r_lower_from_root = rel_vec_mm(RIGHT_KNEE_MM, ROOT_ORIGIN_MM)
    l_parallel_from_lower = rel_vec_mm(LEFT_KNEE_PARALLEL_MM, LEFT_KNEE_MM)
    r_parallel_from_lower = rel_vec_mm(RIGHT_KNEE_PARALLEL_MM, RIGHT_KNEE_MM)
    l_wheel_from_lower = rel_vec_mm(LEFT_WHEEL_MM, LEFT_KNEE_MM)
    r_wheel_from_lower = rel_vec_mm(RIGHT_WHEEL_MM, RIGHT_KNEE_MM)
    l_upper_node_site = rel_vec_mm(LEFT_NODE_UPPER_CONNECT_MM, LEFT_HIP_MM)
    r_upper_node_site = rel_vec_mm(RIGHT_NODE_UPPER_CONNECT_MM, RIGHT_HIP_MM)
    l_node_upper_site = rel_vec_mm(LEFT_NODE_UPPER_CONNECT_MM, LEFT_NODE_PARALLEL_MM)
    r_node_upper_site = rel_vec_mm(RIGHT_NODE_UPPER_CONNECT_MM, RIGHT_NODE_PARALLEL_MM)
    l_node_long_site = rel_vec_mm(LEFT_LONG_NODE_MM, LEFT_NODE_PARALLEL_MM)
    r_node_long_site = rel_vec_mm(RIGHT_LONG_NODE_MM, RIGHT_NODE_PARALLEL_MM)
    l_long_node_site = rel_vec_mm(LEFT_LONG_NODE_MM, LEFT_ACTIVE_LONG_MM)
    r_long_node_site = rel_vec_mm(RIGHT_LONG_NODE_MM, RIGHT_ACTIVE_LONG_MM)
    hip_range = "-3.10 3.10"
    knee_range = "-2.50 2.50"
    active_range = "-2.60 2.60"
    node_range = "-2.60 2.60"
    bar_range = "-2.60 2.60"
    hip_stiffness = 0
    hip_damping = 0.08
    knee_stiffness = 0
    knee_damping = 0.10
    active_stiffness = 0
    node_stiffness = 0
    long_stiffness = 0
    parallel_stiffness = 0
    linkage_damping = 0.08
    loop_connect = 'solref="0.008 1" solimp="0.95 0.99 0.002 0.5 2"'

    return f"""<mujoco model="wheeled_infantry">
  <compiler angle="radian" meshdir="meshes_mj" inertiafromgeom="auto" balanceinertia="true"/>
  <statistic center="0 0 0.25" extent="1.5"/>

  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" iterations="100" ls_iterations="50"/>

  <visual>
    <headlight ambient="0.18 0.18 0.18" diffuse="0.40 0.40 0.40" specular="0.05 0.05 0.05"/>
    <rgba haze="0.15 0.15 0.15 1"/>
  </visual>

  <default>
    <joint damping="0.6" armature="0.01"/>
    <geom margin="0.002" solimp="0.90 0.99 0.002" solref="0.008 1"/>
  </default>

  <asset>
    <!-- Matte black ground material. -->
    <material name="ground_black_mat" rgba="0 0 0 1" reflectance="0"/>
{os.linesep.join(asset_lines)}
  </asset>

  <equality>
    <connect name="left_loop1" site1="left_node_upper_site" site2="left_upper_node_site" {loop_connect}/>
    <connect name="left_loop2" site1="left_node_long_site" site2="left_long_node_site" {loop_connect}/>
    <connect name="right_loop1" site1="right_node_upper_site" site2="right_upper_node_site" {loop_connect}/>
    <connect name="right_loop2" site1="right_node_long_site" site2="right_long_node_site" {loop_connect}/>
  </equality>

  <tendon>
    <!-- Passive spatial tendons measure hip-pivot to wheel-center leg length only. -->
    <spatial name="left_leg_length_tendon">
      <site site="left_leg_length_hip_site"/>
      <site site="left_leg_length_wheel_site"/>
    </spatial>
    <spatial name="right_leg_length_tendon">
      <site site="right_leg_length_hip_site"/>
      <site site="right_leg_length_wheel_site"/>
    </spatial>
  </tendon>

  <contact>
    <!-- Explicit self-contact pairs: wheel/leg travel is limited by the chassis. -->
    <pair name="left_wheel_to_base" geom1="base_collision" geom2="left_wheel_contact"/>
    <pair name="right_wheel_to_base" geom1="base_collision" geom2="right_wheel_contact"/>
    <pair name="left_leg_to_base" geom1="base_collision" geom2="left_lower_leg_collision"/>
    <pair name="right_leg_to_base" geom1="base_collision" geom2="right_lower_leg_collision"/>
    <pair name="left_long_to_base" geom1="base_collision" geom2="left_long_link_collision"/>
    <pair name="right_long_to_base" geom1="base_collision" geom2="right_long_link_collision"/>
    <pair name="left_active_to_lower" geom1="left_active_link_collision" geom2="left_lower_leg_collision"/>
    <pair name="right_active_to_lower" geom1="right_active_link_collision" geom2="right_lower_leg_collision"/>
    <pair name="left_long_to_lower" geom1="left_long_link_collision" geom2="left_lower_leg_collision"/>
    <pair name="right_long_to_lower" geom1="right_long_link_collision" geom2="right_lower_leg_collision"/>
    <pair name="left_long_to_upper" geom1="left_long_link_collision" geom2="left_upper_leg_collision"/>
    <pair name="right_long_to_upper" geom1="right_long_link_collision" geom2="right_upper_leg_collision"/>
    <pair name="left_parallel_to_upper" geom1="left_parallel_link_collision" geom2="left_upper_leg_collision"/>
    <pair name="right_parallel_to_upper" geom1="right_parallel_link_collision" geom2="right_upper_leg_collision"/>
  </contact>

  <worldbody>
    <light name="sun" pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.55 0.55 0.55" specular="0.05 0.05 0.05"/>
    <!-- Static ground: collision group 2 accepts contacts from dynamic group 1. -->
    <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1" material="ground_black_mat" friction="1.10 0.05 0.01" contype="2" conaffinity="1"/>
    <body name="robot" pos="0 0 0.400" quat="0.70710678 0.70710678 0 0">
      <freejoint name="robot_free"/>
      <inertial pos="-0.00509948 0.12135430 -0.00542460" mass="16.62220419" fullinertia="0.40366446 0.60660496 0.37345494 -0.01685508 0.00427268 -0.05125392"/>
      <!-- Dynamic collision group 1 only collides with the ground group 2. -->
      <site name="imu_site" pos="0 0 0" size="0.006" rgba="0.10 0.65 1 0.45"/>
      <!-- Hip-pivot endpoints for direct left/right leg-length measurement. -->
      <site name="left_leg_length_hip_site" pos="{fmt3(l_hip)}" size="0.004" rgba="0.20 1 0.80 0.45"/>
      <site name="right_leg_length_hip_site" pos="{fmt3(r_hip)}" size="0.004" rgba="1 0.55 0.20 0.45"/>
      <geom name="base_contact" type="sphere" pos="0 0 0" size="0.045" mass="0" friction="0.80 0.005 0.0001" contype="4" conaffinity="0" rgba="0 0 0 0"/>
      <!-- Chassis collision proxy used by explicit wheel/leg self-contact pairs. -->
      <geom name="base_collision" type="box" pos="0 0 0" size="0.160 0.070 0.070" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
{build_body_geom_xml("base", indent="      ")}

      <body name="left_upper_leg" pos="{fmt3(l_hip)}">
        <joint name="left_hip_pitch" type="hinge" axis="-1 0 0" range="{hip_range}" damping="{hip_damping}" stiffness="{hip_stiffness}" springref="0"/>
        <geom name="left_upper_leg_collision" type="capsule" fromto="0 0 0 {fmt3(l_active_from_upper)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
        {build_body_geom_xml("l_upper", indent="        ")}
        <site name="left_upper_node_site" pos="{fmt3(l_upper_node_site)}" size="0.006" rgba="1 0.90 0 1"/>
        <body name="left_active_link" pos="{fmt3(l_active_from_upper)}">
          <joint name="left_active_link_pitch" type="hinge" axis="1 0 0" range="{active_range}" damping="{linkage_damping}" stiffness="{active_stiffness}" springref="0"/>
          <geom name="left_active_link_collision" type="capsule" fromto="0 0 0 {fmt3(l_long_from_active)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
          {build_body_geom_xml("l_active", indent="          ")}
          <body name="left_long_link" pos="{fmt3(l_long_from_active)}">
            <joint name="left_long_link_pitch" type="hinge" axis="1 0 0" range="{bar_range}" damping="{linkage_damping}" stiffness="{long_stiffness}" springref="0"/>
            <geom name="left_long_link_collision" type="capsule" fromto="0 0 0 {fmt3(l_long_node_site)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
            {build_body_geom_xml("l_long", indent="            ")}
            <site name="left_long_node_site" pos="{fmt3(l_long_node_site)}" size="0.006" rgba="0.10 1 0.10 1"/>
          </body>
        </body>
      </body>
      <body name="left_lower_leg" pos="{fmt3(l_lower_from_root)}">
          <joint name="left_knee_pitch" type="hinge" axis="1 0 0" range="{knee_range}" damping="{knee_damping}" stiffness="{knee_stiffness}" springref="0"/>
          <!-- Lower-leg sweep proxy for the explicit chassis contact pair. -->
          <geom name="left_lower_leg_collision" type="capsule" fromto="0 0 0 {fmt3(l_wheel_from_lower)}" size="0.022" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
          {build_body_geom_xml("l_lower", indent="          ")}
          <body name="left_parallel_link" pos="{fmt3(l_parallel_from_lower)}">
            <joint name="left_parallel_link_pitch" type="hinge" axis="1 0 0" range="{bar_range}" damping="{linkage_damping}" stiffness="{parallel_stiffness}" springref="0"/>
            <geom name="left_parallel_link_collision" type="capsule" fromto="0 0 0 {fmt3(l_node_from_parallel)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
            {build_body_geom_xml("l_parallel", indent="            ")}
            <body name="left_node_link" pos="{fmt3(l_node_from_parallel)}">
              <joint name="left_node_link_pitch" type="hinge" axis="1 0 0" range="{node_range}" damping="{linkage_damping}" stiffness="{node_stiffness}" springref="0"/>
              <geom name="left_node_link_collision" type="capsule" fromto="0 0 0 {fmt3(l_node_upper_site)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
              {build_body_geom_xml("l_node", indent="              ")}
              <site name="left_node_upper_site" pos="{fmt3(l_node_upper_site)}" size="0.009" rgba="1 0 1 0.55"/>
              <site name="left_node_long_site" pos="{fmt3(l_node_long_site)}" size="0.009" rgba="1 0.25 0.10 0.55"/>
            </body>
          </body>
          <body name="left_wheel" pos="{fmt3(l_wheel_from_lower)}">
            <joint name="left_wheel_spin" type="hinge" axis="1 0 0" damping="0.01"/>
            <site name="left_leg_length_wheel_site" pos="0 0 0" size="0.004" rgba="0.20 1 0.80 0.45"/>
            {build_body_geom_xml("l_wheel_assembly", indent="            ")}
            <geom name="left_wheel_contact" type="cylinder" size="0.060 0.0165" quat="0.70710678 0 0.70710678 0" mass="0" friction="1.10 0.005 0.0001" condim="3" contype="1" conaffinity="2" rgba="0 0 0 0"/>
          </body>
      </body>

      <body name="right_upper_leg" pos="{fmt3(r_hip)}">
        <joint name="right_hip_pitch" type="hinge" axis="1 0 0" range="{hip_range}" damping="{hip_damping}" stiffness="{hip_stiffness}" springref="0"/>
        <geom name="right_upper_leg_collision" type="capsule" fromto="0 0 0 {fmt3(r_active_from_upper)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
        {build_body_geom_xml("r_upper", indent="        ")}
        <site name="right_upper_node_site" pos="{fmt3(r_upper_node_site)}" size="0.006" rgba="1 0.90 0 1"/>
        <body name="right_active_link" pos="{fmt3(r_active_from_upper)}">
          <joint name="right_active_link_pitch" type="hinge" axis="1 0 0" range="{active_range}" damping="{linkage_damping}" stiffness="{active_stiffness}" springref="0"/>
          <geom name="right_active_link_collision" type="capsule" fromto="0 0 0 {fmt3(r_long_from_active)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
          {build_body_geom_xml("r_active", indent="          ")}
          <body name="right_long_link" pos="{fmt3(r_long_from_active)}">
            <joint name="right_long_link_pitch" type="hinge" axis="-1 0 0" range="{bar_range}" damping="{linkage_damping}" stiffness="{long_stiffness}" springref="0"/>
            <geom name="right_long_link_collision" type="capsule" fromto="0 0 0 {fmt3(r_long_node_site)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
            {build_body_geom_xml("r_long", indent="            ")}
            <site name="right_long_node_site" pos="{fmt3(r_long_node_site)}" size="0.006" rgba="0.10 1 0.10 1"/>
          </body>
        </body>
      </body>
      <body name="right_lower_leg" pos="{fmt3(r_lower_from_root)}">
          <joint name="right_knee_pitch" type="hinge" axis="-1 0 0" range="{knee_range}" damping="{knee_damping}" stiffness="{knee_stiffness}" springref="0"/>
          <!-- Lower-leg sweep proxy for the explicit chassis contact pair. -->
          <geom name="right_lower_leg_collision" type="capsule" fromto="0 0 0 {fmt3(r_wheel_from_lower)}" size="0.022" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
          {build_body_geom_xml("r_lower", indent="          ")}
          <body name="right_parallel_link" pos="{fmt3(r_parallel_from_lower)}">
            <joint name="right_parallel_link_pitch" type="hinge" axis="-1 0 0" range="{bar_range}" damping="{linkage_damping}" stiffness="{parallel_stiffness}" springref="0"/>
            <geom name="right_parallel_link_collision" type="capsule" fromto="0 0 0 {fmt3(r_node_from_parallel)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
            {build_body_geom_xml("r_parallel", indent="            ")}
            <body name="right_node_link" pos="{fmt3(r_node_from_parallel)}">
              <joint name="right_node_link_pitch" type="hinge" axis="-1 0 0" range="{node_range}" damping="{linkage_damping}" stiffness="{node_stiffness}" springref="0"/>
              <geom name="right_node_link_collision" type="capsule" fromto="0 0 0 {fmt3(r_node_upper_site)}" size="0.014" mass="0" contype="4" conaffinity="0" rgba="0 0 0 0"/>
              {build_body_geom_xml("r_node", indent="              ")}
              <site name="right_node_upper_site" pos="{fmt3(r_node_upper_site)}" size="0.009" rgba="1 0 1 0.55"/>
              <site name="right_node_long_site" pos="{fmt3(r_node_long_site)}" size="0.009" rgba="1 0.25 0.10 0.55"/>
            </body>
          </body>
          <body name="right_wheel" pos="{fmt3(r_wheel_from_lower)}">
            <joint name="right_wheel_spin" type="hinge" axis="1 0 0" damping="0.01"/>
            <site name="right_leg_length_wheel_site" pos="0 0 0" size="0.004" rgba="1 0.55 0.20 0.45"/>
            {build_body_geom_xml("r_wheel_assembly", indent="            ")}
            <geom name="right_wheel_contact" type="cylinder" size="0.060 0.0165" quat="0.70710678 0 0.70710678 0" mass="0" friction="1.10 0.005 0.0001" condim="3" contype="1" conaffinity="2" rgba="0 0 0 0"/>
          </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <!-- World-frame horizontal displacement and velocity: use X/Y components. -->
    <framepos name="world_horizontal_position_xy" objtype="body" objname="robot"/>
    <framelinvel name="world_horizontal_velocity_xy" objtype="body" objname="robot"/>

    <!-- Left/right wheel rotation about each wheel's local normal axis and derivative. -->
    <jointpos name="left_wheel_angle" joint="left_wheel_spin"/>
    <jointvel name="left_wheel_angular_velocity" joint="left_wheel_spin"/>
    <jointpos name="right_wheel_angle" joint="right_wheel_spin"/>
    <jointvel name="right_wheel_angular_velocity" joint="right_wheel_spin"/>

    <!-- Left/right leg tilt about each hip's local normal axis and derivative. -->
    <jointpos name="left_leg_tilt_angle" joint="left_hip_pitch"/>
    <jointvel name="left_leg_tilt_angular_velocity" joint="left_hip_pitch"/>
    <jointpos name="right_leg_tilt_angle" joint="right_hip_pitch"/>
    <jointvel name="right_leg_tilt_angular_velocity" joint="right_hip_pitch"/>

    <!-- Direct hip-pivot to wheel-center leg length and its derivative. -->
    <tendonpos name="left_leg_length" tendon="left_leg_length_tendon"/>
    <tendonvel name="left_leg_length_velocity" tendon="left_leg_length_tendon"/>
    <tendonpos name="right_leg_length" tendon="right_leg_length_tendon"/>
    <tendonvel name="right_leg_length_velocity" tendon="right_leg_length_tendon"/>

    <!-- Body attitude and angular velocity for world-frame yaw/pitch estimation. -->
    <framequat name="world_body_orientation_quat" objtype="body" objname="robot"/>
    <frameangvel name="body_angular_velocity" objtype="body" objname="robot"/>

    <!-- Body-mounted IMU outputs in the IMU's local coordinate frame. -->
    <gyro name="imu_gyroscope" site="imu_site"/>
    <accelerometer name="imu_linear_accelerometer" site="imu_site"/>
    <velocimeter name="imu_linear_velocity" site="imu_site"/>

    <!-- Left-leg dual hip-motor output torque sensors. -->
    <actuatorfrc name="left_hip_motor_torque" actuator="left_hip_motor"/>
    <actuatorfrc name="left_active_hip_motor_torque" actuator="left_active_hip_motor"/>
    <actuatorfrc name="left_wheel_motor_torque" actuator="left_wheel_motor"/>

    <!-- Right-leg dual hip-motor output torque sensors. -->
    <actuatorfrc name="right_hip_motor_torque" actuator="right_hip_motor"/>
    <actuatorfrc name="right_active_hip_motor_torque" actuator="right_active_hip_motor"/>
    <actuatorfrc name="right_wheel_motor_torque" actuator="right_wheel_motor"/>
  </sensor>

  <actuator>
    <!-- Each leg has two DM-J8009P-2EC hip motors; knee links are passive. -->
    <motor name="left_hip_motor" joint="left_hip_pitch" gear="1" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-40 40"/>
    <motor name="left_active_hip_motor" joint="left_active_link_pitch" gear="1" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-40 40"/>
    <motor name="left_wheel_motor" joint="left_wheel_spin" gear="15.7647058824" ctrllimited="true" ctrlrange="-3 3" forcelimited="true" forcerange="-3 3"/>
    <motor name="right_hip_motor" joint="right_hip_pitch" gear="1" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-40 40"/>
    <motor name="right_active_hip_motor" joint="right_active_link_pitch" gear="1" ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-40 40"/>
    <motor name="right_wheel_motor" joint="right_wheel_spin" gear="15.7647058824" ctrllimited="true" ctrlrange="-3 3" forcelimited="true" forcerange="-3 3"/>
  </actuator>

</mujoco>
"""


if __name__ == "__main__":
    rows = collect_meshes()
    asset_entries_global = []
    body_geom_map_global = defaultdict(list)

    reset_output_dir()

    grouped = defaultdict(list)
    body_mass = defaultdict(float)
    material_mass = defaultdict(float)

    for row in rows:
        material = material_for(row["name"])
        body = special_body_for_center(row["center_mm"])
        if body is None:
            body = to_body_key(body_for(row["name"], row["center_mm"][0], row["center_mm"][1]))
        body = wheel_assembly_body(row, body, material)
        density = DENSITIES[material]
        mass = density * (row["volume_mm3"] * 1e-9)

        row["material"] = material
        row["body"] = body
        row["mass_est"] = mass

        if not should_include_row(row, body):
            continue

        grouped[(body, material)].append(row)
        body_mass[body] += mass
        material_mass[material] += mass

    with open(os.path.join(ROOT, "classification_rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    asset_entries_global, body_geom_map_global = export_grouped_meshes(grouped)

    xml = build_xml(body_geom_map_global)
    with open(XML_OUT, "w", encoding="utf-8") as f:
        f.write(xml)

    summary = {
        "body_mass_kg": dict(sorted(body_mass.items())),
        "material_mass_kg": dict(sorted(material_mass.items())),
        "asset_count": len(asset_entries_global),
    }
    with open(os.path.join(ROOT, "classification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
