from pathlib import Path

import FreeCAD as App
import Import


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Aplus_retrofit"
PARTS = [
    "Aplus_outer_cheek_7075T6.step",
    "Aplus_shoulder_axle_D5_L25_M4.step",
    "Aplus_roller_bushing_ID5_OD6_L10.step",
    "Aplus_roller_bushing_ID5_OD6_L9.step",
    "Aplus_inner_plate_reducer_ID5_OD6_L8.step",
    "Aplus_thrust_washer_ID5p2_OD9_T0p5.step",
    "Aplus_outer_axial_spacer_ID5p2_OD9_T1p3.step",
    "Aplus_standoff_OD8_ID4p3_L12p0.step",
    "Aplus_standoff_fastener_M4x25.step",
    "Aplus_axle_lockwasher_M4.step",
    "Aplus_axle_locknut_M4.step",
    "Aplus_drill_template_common.step",
    "Aplus_40mm_reference_stack.step",
]


for filename in PARTS:
    document = App.newDocument("verify")
    Import.insert(str(OUTPUT_DIR / filename), document.Name)
    document.recompute()
    solids = []
    for item in document.Objects:
        shape = getattr(item, "Shape", None)
        if shape is not None and not shape.isNull():
            solids.extend(shape.Solids)
    if not solids or any(not solid.isValid() or solid.Volume <= 0.0 for solid in solids):
        raise RuntimeError("Invalid STEP geometry: {0}".format(filename))
    print("{0}: {1} solids".format(filename, len(solids)))
    App.closeDocument(document.Name)

print("A+ STEP verification: OK")
