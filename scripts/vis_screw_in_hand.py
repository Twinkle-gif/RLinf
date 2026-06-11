#!/usr/bin/env python3
"""Visualize ScrewInHand initial episode state.

Usage:
    # GUI window (requires display):
    python scripts/vis_screw_in_hand.py

    # Save RGB image to file (headless):
    python scripts/vis_screw_in_hand.py --save init_state.png

    # Choose screw variant:
    python scripts/vis_screw_in_hand.py --type boxnut
    python scripts/vis_screw_in_hand.py --type driver
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize ScrewInHand initial state")
    parser.add_argument("--type", default="trinut", choices=["trinut", "boxnut", "driver"])
    parser.add_argument("--save", default=None, help="Save RGB image to file (e.g. --save init_state.png)")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    import mani_skill  # noqa: F401 – trigger ManiSkill registration
    import rlinf.envs.maniskill  # noqa: F401 – trigger ScrewInHand registration
    import gymnasium as gym
    import os

    # Auto-fix DISPLAY for Docker containers: forward X11 to host via socat
    if not args.save and "DISPLAY" in os.environ:
        cur = os.environ["DISPLAY"]
        if cur.startswith(":") and not os.path.exists(f"/tmp/.X11-unix/X{cur.lstrip(':')}"):
            # Local X socket missing → assume Docker; try host via socat on port 6000
            try:
                import socket
                s = socket.create_connection(("172.17.0.1", 6000), timeout=1)
                s.close()
                os.environ["DISPLAY"] = "172.17.0.1:0"
                print(f"DISPLAY auto-redirected: {cur} → 172.17.0.1:0")
            except Exception:
                pass

    env_id = f"ScrewInHand-{args.type}-v1"
    print(f"Creating env: {env_id}")

    # Use GUI by default; fall back to rgb_array only when --save is given
    render_mode = "rgb_array" if args.save else "human"
    env = gym.make(
        env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=render_mode,
        sim_backend="cpu",  # CPU for easy local vis
    )

    obs, info = env.reset()
    print(f"Reset done. obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"Info keys: {[k for k in info.keys() if not k.startswith('_')]}")

    # Print key state
    base_env = env.unwrapped

    # --- Add XYZ coordinate axes near palm for visibility ---
    import sapien
    scene = base_env.scene
    axis_len = 0.10   # 10 cm (larger for visibility)
    axis_radius = 0.004  # 4 mm (thicker)
    colors = {
        "x": [1, 0, 0, 1],   # Red
        "y": [0, 1, 0, 1],   # Green
        "z": [0, 0, 1, 1],   # Blue
    }
    # Place axes at palm position so they're always visible near the hand
    palm_p = base_env.agent.palm_link.pose.p[0].cpu().numpy()
    palm_q = base_env.agent.palm_link.pose.q[0].cpu().numpy()

    # X-axis: cylinder along X
    b = scene.create_actor_builder()
    b.add_cylinder_visual(
        radius=axis_radius, half_length=axis_len / 2,
        material=sapien.render.RenderMaterial(base_color=colors["x"]),
    )
    b.set_initial_pose(sapien.Pose(
        p=[axis_len / 2, 0, 0],
        q=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
    ))
    axis_x = b.build_static(name="axis_x")

    # Y-axis: rotate 90° around X to align with Y
    b = scene.create_actor_builder()
    b.add_cylinder_visual(
        radius=axis_radius, half_length=axis_len / 2,
        material=sapien.render.RenderMaterial(base_color=colors["y"]),
    )
    b.set_initial_pose(sapien.Pose(
        p=[0, axis_len / 2, 0],
        q=[np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0],
    ))
    axis_y = b.build_static(name="axis_y")

    # Z-axis: cylinder default is Z, no rotation needed
    b = scene.create_actor_builder()
    b.add_cylinder_visual(
        radius=axis_radius, half_length=axis_len / 2,
        material=sapien.render.RenderMaterial(base_color=colors["z"]),
    )
    b.set_initial_pose(sapien.Pose(p=[0, 0, axis_len / 2]))
    axis_z = b.build_static(name="axis_z")

    print("Added XYZ axes (R=X, G=Y, B=Z) at world origin")
    print(f"\n--- Initial state ---")
    print(f"  Palm position:  {base_env.agent.palm_link.pose.p[0].cpu().numpy()}")
    print(f"  Palm quaternion (wxyz): {base_env.agent.palm_link.pose.q[0].cpu().numpy()}")
    print(f"  Nut position:   {base_env.nut_link.pose.p[0].cpu().numpy()}")
    print(f"  Nut z:          {base_env.nut_link.pose.p[0, 2].item():.4f}")
    print(f"  Hand init qpos (first 4): {base_env.agent.robot.qpos[0, :4].cpu().numpy()}")
    print(f"  Screw qpos:     {base_env.screw.qpos[0].item():.4f}")

    if args.save:
        # Headless: render to rgb_array and save
        img = env.render()
        from PIL import Image as PILImage
        if isinstance(img, dict):
            for cam_name, cam_img in img.items():
                arr = cam_img[0].cpu().numpy() if hasattr(cam_img, 'cpu') else cam_img
                save_path = args.save.replace(".png", f"_{cam_name}.png") if len(img) > 1 else args.save
                PILImage.fromarray(arr).save(save_path)
                print(f"Saved: {save_path}")
        else:
            arr = img.cpu().numpy() if hasattr(img, 'cpu') else np.asarray(img)
            arr = arr.squeeze()
            PILImage.fromarray(arr).save(args.save)
            print(f"Saved: {args.save}")
    else:
        # Interactive: open GUI viewer and keep it alive
        viewer = env.render()
        print("GUI viewer opened. Close the window or press Ctrl+C to exit.")
        try:
            while not viewer.window.should_close:
                viewer.render()
        except KeyboardInterrupt:
            pass

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()