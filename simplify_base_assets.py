"""Replace only the base visual meshes while preserving the current leg XML tree."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import build_wheeled_infantry as build


ROOT = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(ROOT, "wheeled_infantry.xml")
ROWS_PATH = os.path.join(ROOT, "classification_rows.json")
OUT_DIR = os.path.join(ROOT, "meshes_mj")
MAX_FACES = 200000

KEEP_KEYWORDS = (
    "铝管框架",
    "铝管长嵌件",
    "铝管短嵌件",
    "横梁嵌件",
    "电机梁嵌件",
    "底盘设备固定板",
    "GM6020固定板",
    "髋关节电机固定板",
    "髋关节碗组轴承座",
    "髋关节轴上压紧盖",
    "髋关节轴下压紧盖",
    "装甲",
    "云台",
    "鹅颈",
    "发射头",
    "枪管",
    "设备仓",
)
EXCLUDE_KEYWORDS = (
    "电机",
    "轴承",
    "螺",
    "螺母",
    "垫片",
    "相机",
    "图传",
    "PCB",
    "C板",
    "电调",
    "电脑",
    "电源",
    "电池",
    "主控",
    "测速",
    "测距",
    "灯条",
    "导轮",
    "同步带",
    "同步轮",
    "滑环",
    "限位",
    "传动轴",
)
MATERIAL_STYLE = {
    "aluminum": ("2700.0", "0.74 0.76 0.79 1"),
    "carbon": ("1650.0", "0.08 0.08 0.09 1"),
    "plastic": ("1240.0", "0.16 0.16 0.18 1"),
}


def keep_base_row(row: dict) -> bool:
    name = row["name"]
    if row["body"] != "base" or any(key in name for key in EXCLUDE_KEYWORDS):
        return False
    return any(key in name for key in KEEP_KEYWORDS)


def export_base_meshes(rows: list[dict]) -> list[tuple[str, str, str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        material = row["material"]
        if material in MATERIAL_STYLE:
            grouped[material].append(row)

    assets = []
    for material, items in sorted(grouped.items()):
        chunks: list[list[tuple]] = []
        current: list[tuple] = []
        for item in items:
            _, triangles = build.read_binary_stl(item["path"])
            shifted = [
                tuple(tuple(point[i] - build.ROOT_ORIGIN_MM[i] for i in range(3)) for point in triangle)
                for triangle in triangles
            ]
            for triangle in shifted:
                if len(current) == MAX_FACES:
                    chunks.append(current)
                    current = []
                current.append(triangle)
        if current:
            chunks.append(current)

        for index, triangles in enumerate(chunks, start=1):
            suffix = f"_part{index}" if len(chunks) > 1 else ""
            mesh_name = f"base_visual_{material}{suffix}"
            mesh_file = f"{mesh_name}.stl"
            build.write_binary_stl(os.path.join(OUT_DIR, mesh_file), triangles)
            assets.append((mesh_name, mesh_file, material))
    return assets


def update_xml(assets: list[tuple[str, str, str]]) -> None:
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    asset = root.find("asset")
    robot = root.find("./worldbody/body[@name='robot']")
    if asset is None or robot is None:
        raise RuntimeError("Expected asset section and robot body are missing")

    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").startswith("base_"):
            asset.remove(mesh)
    for geom in list(robot.findall("geom")):
        if geom.get("mesh", "").startswith("base_"):
            robot.remove(geom)

    for mesh_name, mesh_file, material in assets:
        ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file, scale="0.001 0.001 0.001")
    for mesh_name, _, material in reversed(assets):
        density, rgba = MATERIAL_STYLE[material]
        robot.insert(
            0,
            ET.Element(
                "geom",
                type="mesh",
                mesh=mesh_name,
                density=density,
                rgba=rgba,
                contype="0",
                conaffinity="0",
                group="0",
            ),
        )

    ET.indent(tree, space="  ")
    tree.write(XML_PATH, encoding="utf-8", xml_declaration=False)


def main() -> None:
    with open(ROWS_PATH, encoding="utf-8") as file:
        rows = json.load(file)
    selected = [row for row in rows if keep_base_row(row)]
    assets = export_base_meshes(selected)
    update_xml(assets)
    print(f"Kept {len(selected)} base source meshes as {len(assets)} loaded mesh assets.")


if __name__ == "__main__":
    main()
