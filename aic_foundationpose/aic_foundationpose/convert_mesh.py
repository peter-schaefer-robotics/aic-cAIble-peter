"""Convert NIC Card GLB mesh to OBJ for FoundationPose.

Usage (after colcon build):
  ros2 run aic_foundationpose convert_nic_mesh

Or directly:
  python3 convert_mesh.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


_DEFAULT_GLB = Path(__file__).parents[2] / "aic_assets" / "models" / "NIC Card" / "nic_card_visual.glb"
_DEFAULT_OUT = Path(__file__).parent.parent / "meshes" / "nic_card.obj"


def convert(glb_path: Path, out_path: Path) -> None:
    print(f"Loading: {glb_path}")
    scene_or_mesh = trimesh.load(str(glb_path))

    # GLB files often load as a Scene; merge all geometry into one mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("No triangle meshes found in GLB scene.")
        mesh = trimesh.util.concatenate(meshes)
        print(f"Merged {len(meshes)} scene geometries.")
    else:
        mesh = scene_or_mesh

    # Centre at origin (FoundationPose convention)
    mesh.apply_translation(-mesh.centroid)

    # Report dimensions so the user can sanity-check scale (should be metres)
    extents = mesh.extents
    print(f"Mesh extents (x,y,z): {extents}")
    print(f"  → If extents look like millimetres (> 10), multiply by 0.001")

    # Check if scale looks like millimetres and warn
    if np.max(extents) > 0.5:
        print("WARNING: mesh extents suggest millimetre units. Rescaling by 0.001 …")
        mesh.apply_scale(0.001)
        print(f"  New extents: {mesh.extents}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"Saved: {out_path}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NIC Card GLB to OBJ")
    parser.add_argument("--input", default=str(_DEFAULT_GLB), help="Input GLB path")
    parser.add_argument("--output", default=str(_DEFAULT_OUT), help="Output OBJ path")
    args = parser.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
