import csv
import json
import math
import os
from pathlib import Path

import FreeCAD as App
import Import
import Part


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "guide_wheel_a_plus.yaml"
OUTPUT_DIR = ROOT / "Aplus_retrofit"


def cylinder_x(radius, length, x_start, y=0.0, z=0.0):
    return Part.makeCylinder(radius, length, App.Vector(x_start, y, z), App.Vector(1, 0, 0))


def plate_bar(a, b, width, x_start, thickness):
    ay, az = a
    by, bz = b
    dy = by - ay
    dz = bz - az
    magnitude = (dy * dy + dz * dz) ** 0.5
    if magnitude <= 0.0:
        raise ValueError("Bridge endpoints must be distinct.")
    ny = -dz / magnitude * width / 2.0
    nz = dy / magnitude * width / 2.0
    points = [
        App.Vector(x_start, ay + ny, az + nz),
        App.Vector(x_start, by + ny, bz + nz),
        App.Vector(x_start, by - ny, bz - nz),
        App.Vector(x_start, ay - ny, az - nz),
    ]
    wire = Part.makePolygon(points + [points[0]])
    return Part.Face(wire).extrude(App.Vector(thickness, 0, 0))


def make_outer_cheek(name, offsets, cfg):
    fork = cfg["fork"]
    thickness = fork["outer_cheek_thickness"]
    axle = (0.0, 0.0)
    shape = cylinder_x(fork["axle_boss_radius"], thickness, 0.0)
    for offset in offsets:
        point = (float(offset[0]), float(offset[1]))
        shape = shape.fuse(cylinder_x(fork["mount_boss_radius"], thickness, 0.0, point[0], point[1]))
        shape = shape.fuse(plate_bar(axle, point, fork["bridge_width"], 0.0, thickness))
    shape = shape.removeSplitter()
    shape = shape.cut(cylinder_x(fork["axle_clearance_diameter"] / 2.0, thickness + 2.0, -1.0))
    for offset in offsets:
        shape = shape.cut(
            cylinder_x(
                fork["mount_clearance_diameter"] / 2.0,
                thickness + 2.0,
                -1.0,
                float(offset[0]),
                float(offset[1]),
            )
        )
    return name, shape.removeSplitter()


def make_shoulder_axle(cfg):
    axle = cfg["axle"]
    head = cylinder_x(axle["head_diameter"] / 2.0, axle["head_thickness"], -axle["head_thickness"])
    shoulder = cylinder_x(axle["shoulder_diameter"] / 2.0, axle["shoulder_length"], 0.0)
    thread = cylinder_x(axle["thread_diameter"] / 2.0, axle["thread_length"], axle["shoulder_length"])
    return "Aplus_shoulder_axle_D5_L25_M4", head.fuse(shoulder).fuse(thread).removeSplitter()


def make_bushing(name, inner_diameter, outer_diameter, length):
    shape = cylinder_x(outer_diameter / 2.0, length, 0.0)
    shape = shape.cut(cylinder_x(inner_diameter / 2.0, length + 2.0, -1.0))
    return name, shape.removeSplitter()


def make_washer(cfg):
    washer = cfg["thrust_washer"]
    return make_bushing(
        "Aplus_thrust_washer_ID5p2_OD9_T0p5",
        washer["inner_diameter"],
        washer["outer_diameter"],
        washer["thickness"],
    )


def make_outer_axial_spacer(cfg):
    spacer = cfg["outer_axial_spacer"]
    return make_bushing(
        "Aplus_outer_axial_spacer_ID5p2_OD9_T1p3",
        spacer["inner_diameter"],
        spacer["outer_diameter"],
        spacer["thickness"],
    )


def make_standoff(cfg):
    fork = cfg["fork"]
    return make_bushing(
        "Aplus_standoff_OD8_ID4p3_L12p0",
        fork["mount_clearance_diameter"],
        fork["standoff_outer_diameter"],
        fork["standoff_length"],
    )


def make_standoff_fastener(cfg):
    fastener = cfg["standoff_fastener"]
    head = cylinder_x(
        fastener["head_diameter"] / 2.0,
        fastener["head_thickness"],
        -fastener["head_thickness"],
    )
    shaft = cylinder_x(fastener["shaft_diameter"] / 2.0, fastener["shaft_length"], 0.0)
    return "Aplus_standoff_fastener_M4x25", head.fuse(shaft).removeSplitter()


def make_hex_prism_x(across_flats, thickness, x_start):
    radius = across_flats / math.sqrt(3.0)
    points = []
    for index in range(6):
        angle = math.pi / 6.0 + index * math.pi / 3.0
        points.append(App.Vector(x_start, radius * math.cos(angle), radius * math.sin(angle)))
    wire = Part.makePolygon(points + [points[0]])
    return Part.Face(wire).extrude(App.Vector(thickness, 0.0, 0.0))


def make_axle_lockwasher(cfg):
    lock = cfg["axle_lock"]
    return make_bushing(
        "Aplus_axle_lockwasher_M4",
        lock["washer_inner_diameter"],
        lock["washer_outer_diameter"],
        lock["washer_thickness"],
    )


def make_axle_locknut(cfg):
    lock = cfg["axle_lock"]
    shape = make_hex_prism_x(lock["nut_across_flats"], lock["nut_thickness"], 0.0)
    shape = shape.cut(cylinder_x(lock["nut_clearance_diameter"] / 2.0, lock["nut_thickness"] + 2.0, -1.0))
    return "Aplus_axle_locknut_M4", shape.removeSplitter()


def make_drill_template(name, offsets, cfg):
    fork = cfg["fork"]
    thickness = 1.0
    axle = (0.0, 0.0)
    shape = cylinder_x(fork["axle_boss_radius"], thickness, 0.0)
    for offset in offsets:
        point = (float(offset[0]), float(offset[1]))
        shape = shape.fuse(cylinder_x(fork["mount_boss_radius"], thickness, 0.0, point[0], point[1]))
        shape = shape.fuse(plate_bar(axle, point, fork["bridge_width"], 0.0, thickness))
    shape = shape.removeSplitter()
    shape = shape.cut(cylinder_x(3.05, thickness + 2.0, -1.0))
    for offset in offsets:
        shape = shape.cut(
            cylinder_x(
                fork["inner_plate_drill"]["tap_drill_diameter"] / 2.0,
                thickness + 2.0,
                -1.0,
                float(offset[0]),
                float(offset[1]),
            )
        )
    return name, shape.removeSplitter()


def make_reference_stack(cfg, cheek_shape):
    stack = cfg["assembly_stack"]
    axle = cfg["axle"]
    bushings = cfg["bushings"]
    washer = cfg["thrust_washer"]
    outer_spacer = cfg["outer_axial_spacer"]
    fork = cfg["fork"]
    plate_thickness = stack["inner_plate_thickness"]
    inner_washer_start = -washer["thickness"]
    wheel_start = 0.0
    outer_washer_start = wheel_start + cfg["scope"]["roller_width"]
    outer_spacer_start = outer_washer_start + washer["thickness"] + stack["axial_endplay"]
    cheek_start = outer_spacer_start + outer_spacer["thickness"]
    objects = []

    inner_plate_proxy = Part.makeBox(
        plate_thickness, 60.0, 70.0, App.Vector(-plate_thickness, -30.0, -35.0)
    )
    inner_plate_proxy = inner_plate_proxy.cut(cylinder_x(3.0, plate_thickness + 2.0, -plate_thickness - 1.0))
    inner_plate_proxy = inner_plate_proxy.cut(
        cylinder_x(
            cfg["fork"]["inner_thrust_recess"]["diameter"] / 2.0,
            cfg["fork"]["inner_thrust_recess"]["depth"],
            -cfg["fork"]["inner_thrust_recess"]["depth"],
        )
    )
    objects.append(("reference_inner_mount_plate", inner_plate_proxy))

    reducer = make_bushing(
        "reference_inner_reducer",
        bushings["inner_plate_id"],
        bushings["inner_plate_od"],
        bushings["inner_plate_length"],
    )[1].translated(App.Vector(-plate_thickness, 0.0, 0.0))
    objects.append(("reference_inner_reducer", reducer))

    inner_washer = make_bushing(
        "reference_inner_thrust_washer",
        washer["inner_diameter"],
        washer["outer_diameter"],
        washer["thickness"],
    )[1].translated(App.Vector(inner_washer_start, 0.0, 0.0))
    objects.append(("reference_inner_thrust_washer", inner_washer))

    wheel_proxy = cylinder_x(
        cfg["scope"]["roller_outer_diameter"] / 2.0,
        cfg["scope"]["roller_width"],
        wheel_start,
    )
    wheel_proxy = wheel_proxy.cut(cylinder_x(3.0, cfg["scope"]["roller_width"] + 2.0, wheel_start - 1.0))
    objects.append(("reference_existing_40mm_guide_roller", wheel_proxy))

    roller_bushing = make_bushing(
        "reference_roller_bushing",
        bushings["roller_id"],
        bushings["roller_od"],
        bushings["roller_40_length"],
    )[1].translated(App.Vector(wheel_start, 0.0, 0.0))
    objects.append(("reference_roller_bushing", roller_bushing))

    outer_washer = make_bushing(
        "reference_outer_thrust_washer",
        washer["inner_diameter"],
        washer["outer_diameter"],
        washer["thickness"],
    )[1].translated(App.Vector(outer_washer_start, 0.0, 0.0))
    objects.append(("reference_outer_thrust_washer", outer_washer))

    outer_axial_spacer = make_outer_axial_spacer(cfg)[1].translated(
        App.Vector(outer_spacer_start, 0.0, 0.0)
    )
    objects.append(("reference_outer_axial_spacer", outer_axial_spacer))

    objects.append(("reference_outer_cheek", cheek_shape.translated(App.Vector(cheek_start, 0.0, 0.0))))
    objects.append(("reference_shoulder_axle", make_shoulder_axle(cfg)[1].translated(App.Vector(-plate_thickness, 0.0, 0.0))))
    lockwasher_start = cheek_start + fork["outer_cheek_thickness"]
    locknut_start = lockwasher_start + cfg["axle_lock"]["washer_thickness"]
    objects.append(("reference_axle_lockwasher", make_axle_lockwasher(cfg)[1].translated(App.Vector(lockwasher_start, 0.0, 0.0))))
    objects.append(("reference_axle_locknut", make_axle_locknut(cfg)[1].translated(App.Vector(locknut_start, 0.0, 0.0))))

    for index, offset in enumerate(fork["mount_offsets_xy_local"], start=1):
        standoff = make_standoff(cfg)[1].translated(
            App.Vector(0.0, float(offset[0]), float(offset[1]))
        )
        objects.append(("reference_standoff_{0}".format(index), standoff))

    return objects


def add_feature(document, name, shape, material):
    if not shape.isValid() or shape.Volume <= 0.0:
        raise RuntimeError("Invalid generated shape: {0}".format(name))
    feature = document.addObject("PartDesign::Feature", name)
    feature.Label = name
    feature.Shape = shape
    feature.addProperty("App::PropertyString", "MaterialSpec")
    feature.MaterialSpec = material
    return feature


def main():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = App.newDocument("guide_wheel_Aplus_retrofit")
    generated = []

    cheek_name, cheek_shape = make_outer_cheek(
        "Aplus_outer_cheek_7075T6",
        cfg["fork"]["mount_offsets_xy_local"],
        cfg,
    )
    generated.extend([
        (cheek_name, cheek_shape, cfg["materials"]["outer_cheek"]),
        (make_shoulder_axle(cfg)[0], make_shoulder_axle(cfg)[1], cfg["materials"]["shoulder_axle"]),
        (make_bushing("Aplus_roller_bushing_ID5_OD6_L10", 5.0, 6.0, 10.0)[0], make_bushing("Aplus_roller_bushing_ID5_OD6_L10", 5.0, 6.0, 10.0)[1], cfg["materials"]["roller_bushing"]),
        (make_bushing("Aplus_roller_bushing_ID5_OD6_L9", 5.0, 6.0, 9.0)[0], make_bushing("Aplus_roller_bushing_ID5_OD6_L9", 5.0, 6.0, 9.0)[1], cfg["materials"]["roller_bushing"]),
        (make_bushing("Aplus_inner_plate_reducer_ID5_OD6_L8", 5.0, 6.0, 8.0)[0], make_bushing("Aplus_inner_plate_reducer_ID5_OD6_L8", 5.0, 6.0, 8.0)[1], cfg["materials"]["inner_plate_reducer"]),
        (make_washer(cfg)[0], make_washer(cfg)[1], cfg["materials"]["thrust_washer"]),
        (make_outer_axial_spacer(cfg)[0], make_outer_axial_spacer(cfg)[1], cfg["materials"]["outer_axial_spacer"]),
        (make_standoff(cfg)[0], make_standoff(cfg)[1], cfg["materials"]["standoff"]),
        (make_standoff_fastener(cfg)[0], make_standoff_fastener(cfg)[1], cfg["standoff_fastener"]["representation"]),
        (make_axle_lockwasher(cfg)[0], make_axle_lockwasher(cfg)[1], "hardened stainless steel"),
        (make_axle_locknut(cfg)[0], make_axle_locknut(cfg)[1], cfg["axle_lock"]["representation"]),
        (make_drill_template("Aplus_drill_template_common", cfg["fork"]["mount_offsets_xy_local"], cfg)[0], make_drill_template("Aplus_drill_template_common", cfg["fork"]["mount_offsets_xy_local"], cfg)[1], "drill template"),
    ])

    features = []
    for name, shape, material in generated:
        feature = add_feature(document, name, shape, material)
        features.append(feature)
        Import.export([feature], str(OUTPUT_DIR / (name + ".step")))

    reference_features = []
    for name, shape in make_reference_stack(cfg, cheek_shape):
        reference_features.append(add_feature(document, name, shape, "reference only"))
    document.recompute()
    Import.export(reference_features, str(OUTPUT_DIR / "Aplus_40mm_reference_stack.step"))
    document.saveAs(str(OUTPUT_DIR / "Aplus_guide_wheel_retrofit.FCStd"))

    with (OUTPUT_DIR / "BOM.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["part", "quantity_for_4_impact_rollers", "material_or_spec"])
        writer.writerow([cheek_name, 4, cfg["materials"]["outer_cheek"]])
        writer.writerow(["Aplus_shoulder_axle_D5_L25_M4", 4, cfg["materials"]["shoulder_axle"]])
        writer.writerow(["Aplus_roller_bushing_ID5_OD6_L10", 4, cfg["materials"]["roller_bushing"]])
        writer.writerow(["Aplus_inner_plate_reducer_ID5_OD6_L8", 4, cfg["materials"]["inner_plate_reducer"]])
        writer.writerow(["Aplus_thrust_washer_ID5p2_OD9_T0p5", 8, cfg["materials"]["thrust_washer"]])
        writer.writerow(["Aplus_outer_axial_spacer_ID5p2_OD9_T1p3", 4, cfg["materials"]["outer_axial_spacer"]])
        writer.writerow(["Aplus_standoff_OD8_ID4p3_L12p0", 8, cfg["materials"]["standoff"]])
        writer.writerow(["Aplus_standoff_fastener_M4x25", 8, cfg["standoff_fastener"]["representation"]])
        writer.writerow(["Aplus_axle_lockwasher_M4", 4, "hardened stainless steel"])
        writer.writerow(["Aplus_axle_locknut_M4", 4, cfg["axle_lock"]["representation"]])

    manifest = {
        "config": "../configs/guide_wheel_a_plus.yaml",
        "output": "Aplus_retrofit",
        "generated_parts": [feature.Label for feature in features],
        "reference_assembly": "Aplus_40mm_reference_stack.step",
        "safety_note": "Do not assemble or drill from this package until the live SolidWorks interface probe confirms the roller and mount-plate coordinate frames.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Generated {0} production STEP parts and one reference stack in {1}".format(len(features), OUTPUT_DIR))


main()
