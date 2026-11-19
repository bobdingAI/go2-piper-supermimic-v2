"""Fetch just the Go2 + PiPER MJCF and its meshes from a pinned revision."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

REVISION = "4e12109d1eccb1d9da2e4a35f189fabea3f13677"
UPSTREAM = f"https://raw.githubusercontent.com/zzzJie-Robot/LeggedManip_Lab/{REVISION}"
SOURCE = "mujoco/robots/go2_piper"
DESTINATION = Path(__file__).resolve().parent / "vendor" / "go2_piper"


def fetch(path: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        ["curl", "-fLsS", "--connect-timeout", "10", "--max-time", "60", "--retry", "2",
         f"{UPSTREAM}/{path}", "-o", str(temporary)],
        check=True,
    )
    temporary.replace(destination)


def main() -> None:
    model = DESTINATION / "go2piper.xml"
    fetch(f"{SOURCE}/{model.name}", model)
    mesh_files = sorted({mesh.attrib["file"] for mesh in ET.parse(model).findall("./asset/mesh")})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [pool.submit(fetch, f"{SOURCE}/{name}", DESTINATION / name) for name in mesh_files]
        results.append(pool.submit(fetch, "LICENSE", DESTINATION / "LICENSE"))
        for result in results:
            result.result()
    print(f"Loaded the upstream asset files for {REVISION} ({len(mesh_files)} meshes).")


if __name__ == "__main__":
    main()
