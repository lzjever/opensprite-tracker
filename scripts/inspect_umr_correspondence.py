#!/usr/bin/env python3
"""Independent semantic and binding QA of newly learned Geno/Sprite slots.

This reads native UMR outputs and calls its unchanged mesh-binding routine.
FBX skin weights supply anatomical QA labels, never optimizer constraints.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"third_party/umr/scripts"))
import mujoco
import numpy as np
from scipy.spatial import cKDTree
import nr_source
import smpl_surface_retarget_common as common
from mujoco_geom_surface import geom_local_mesh, surface_geom_ids
from opensprite_tracker.robot import ROBOT_XML

GROUPS = ("head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg", "unknown")


def source_group(name):
    if "Head" in name or "Neck" in name:
        return "head"
    if "Hips" in name or "Spine" in name:
        return "trunk"
    for prefix, side in (("Left", "left"), ("Right", "right")):
        if name.startswith(prefix):
            return side+"_leg" if any(s in name for s in ("Leg", "Foot", "Toe")) else side+"_arm"
    return "unknown"


def robot_group(name):
    if name.startswith("head_"):
        return "head"
    if name.startswith(("pelvis", "waist")):
        return "trunk"
    side = name.split("_")[0]
    return side+"_leg" if any(s in name for s in ("hip_", "knee_", "ankle_")) else side+"_arm"


def main(output):
    base = ROOT/"assets/umr_correspondence/geno_opensprite0825"
    with np.load(base/"dataset.npz", allow_pickle=False) as archive:
        dataset = {k:archive[k] for k in ("names", "mesh_vertices_0", "mesh_faces_0", "mesh_vertices_1", "mesh_faces_1")}
    with np.load(base/"training/correspondence_slots_final.npz", allow_pickle=False) as archive:
        slots = {k:archive[k] for k in ("names", "reconstructed_slots", "root_offsets", "normalization_scales")}
    assert list(dataset["names"]) == list(slots["names"]) == ["geno", "opensprite0825"]
    bindings=[]
    for i in range(2):
        bindings.append(common.bind_points_to_mesh(slots["reconstructed_slots"][i],
            dataset[f"mesh_vertices_{i}"], dataset[f"mesh_faces_{i}"], nearest_vertex_k=24))
    source, target = bindings
    asset = nr_source.load_fbx_asset(ROOT/"assets/geno/Geno.fbx")
    rest = asset["template_vertices_world"]*.01 - slots["root_offsets"][0]
    distances, original_vertex = cKDTree(rest).query(dataset["mesh_vertices_0"])
    assert distances.max() < 1e-5
    face_vertices = original_vertex[dataset["mesh_faces_0"][source["face_ids"]]]
    bone_ids = asset["skin_indices"][face_vertices]
    weights = asset["skin_weights"][face_vertices]*source["bary"][:,:,None]
    bone_mass=np.zeros((len(face_vertices),len(asset["bone_names"])))
    np.add.at(bone_mass, (np.arange(len(face_vertices))[:,None,None], bone_ids), weights)
    bone_names=np.asarray(asset["bone_match_names"], str)[bone_mass.argmax(1)]
    source_labels=np.array([source_group(n) for n in bone_names])
    m=mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    geoms=surface_geom_ids(m, "visual")
    counts=[len(geom_local_mesh(m,int(g))[1]) for g in geoms]
    ends=np.cumsum(counts)
    face_geoms=geoms[np.searchsorted(ends,target["face_ids"],side="right")]
    body_names=np.array([m.body(int(m.geom_bodyid[g])).name for g in face_geoms])
    target_labels=np.array([robot_group(n) for n in body_names])
    confusion={s:{t:int(np.sum((source_labels==s)&(target_labels==t))) for t in GROUPS}
               for s in GROUPS}
    known=source_labels!="unknown"
    wrong_side=((np.char.startswith(source_labels,"left")&np.char.startswith(target_labels,"right"))|
                (np.char.startswith(source_labels,"right")&np.char.startswith(target_labels,"left")))
    bound=np.stack([b["closest_points"] for b in bindings])
    normals=np.stack([b["closest_normals"] for b in bindings])
    scales=slots["normalization_scales"]
    normalized=bound/scales[:,None,None]
    axis_correlation=[float(np.corrcoef(normalized[0,:,i],normalized[1,:,i])[0,1]) for i in range(3)]
    normal_dot=np.einsum("ij,ij->i",normals[0],normals[1])
    result=dict(slot_count=len(source_labels), names=slots["names"].tolist(),
                wrong_left_right_count=int(wrong_side.sum()),
                coarse_anatomical_match_fraction=float(np.mean(source_labels[known]==target_labels[known])),
                unknown_source_bones=sorted(set(bone_names[~known].tolist())),
                source_to_target_part_counts=confusion,
                normalized_axis_correlations_xyz=axis_correlation,
                initial_normal_dot_median=float(np.median(normal_dot)),
                initial_normal_opposed_fraction=float(np.mean(normal_dot < -.5)),
                normal_note="Native tpose_offset preserves each bind-time normal offset; raw source/robot normal equality is not assumed.",
                bindings={})
    for i,(name,binding) in enumerate(zip(slots["names"],bindings)):
        length=np.linalg.norm(binding["closest_normals"],axis=1)
        result["bindings"][str(name)]=dict(
            point_projection_mean_m=float(np.mean(binding["errors"])),
            point_projection_p95_m=float(np.quantile(binding["errors"],.95)),
            point_projection_max_m=float(np.max(binding["errors"])),
            unit_normal_max_error=float(np.max(abs(length-1))),
            zero_normal_count=int(np.sum(length<.5)),
            barycentric_sum_max_error=float(np.max(abs(binding["bary"].sum(1)-1))))
    result["robot_soles"]={}
    for side in ("left","right"):
        foot=body_names==f"{side}_ankle_roll_link"
        near_bottom=bound[1,:,1] < dataset["mesh_vertices_1"][:,1].min()+.006
        down=normals[1,:,1]<-.7
        selected=foot&near_bottom&down
        result["robot_soles"][side]=dict(total_foot_slots=int(foot.sum()),
            downward_sole_slots=int(selected.sum()),
            mapped_from_source_foot_bones=int(np.sum(selected & np.array(["Foot" in n or "Toe" in n for n in bone_names]))),
            mapped_source_bones=sorted(set(bone_names[selected].tolist())))
    output.mkdir(parents=True,exist_ok=True)
    (output/"correspondence_review.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(output/"correspondence_review_points.npz",points=normalized,
        normals=normals, source_groups=source_labels,target_groups=target_labels,
        source_bones=bone_names,target_bodies=body_names,wrong_side=wrong_side)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/umr/correspondence-review")
    main(parser.parse_args().output)
