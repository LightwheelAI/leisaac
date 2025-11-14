# Run LeIsaac Instantly in the Cloud with NVIDIA Brev

The fastest way to get started with LeIsaac — you don't need a high-performance GPU, just a web browser.

Open a web browser and navigate to this [link](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-35P96N3pyzVDW3Xlohy7X2TuLCX). After the deployment is complete, click the link for port 80 (HTTP) to open Visual Studio Code Server. The default password is `password`.

Quick install:
```bash
cd leisaac
pip install -e source/leisaac
```

Our four open-source scenarios have been pre-installed and can be started using the following command:
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --kit_args="--no-window --enable omni.kit.livestream.webrtc"
```

Then you can open a new browser tab to view the UI. In this tab, paste the same address as the Visual Studio Code server, changing the end of the URL to `/viewer`.

:::info[Example]
If VS Code Server is at `ec2.something.amazonaws.com`, then the UI can be accessed at `ec2.something.amazonaws.com/viewer`.
:::

After a few seconds you should see the UI in the viewer tab. The first launch may take much longer as shaders are cached.

Here is our demo video:

<video
  controls
  src="https://github.com/user-attachments/assets/35228eb4-6e2f-4dc1-b066-fff616ca4505"
  style={{ width: '100%', maxWidth: '960px', borderRadius: '8px' }}
/>
