Two new projects are now live in the tenant:
tlv-pe-hw2 - DDP assignment
tlv-pe-hw3 - Quantization and Benchmarking assignment

HW3 - a notebook service is enough to get started.
HW2 (DDP) - this builds on MLOps HW1, so you already know the drill: spin up a cluster, connect to a registry, connect to SkyPilot. 
Keep boot disks small - 200GB is enough

Re HW2 - A few things to keep in mind:

This assignment requires 4 nodes - do not leave your k8s cluster running idle. You'll block your peers. No slot system this time, so it's on you to be mindful.
SA and SkyPilot service are already set up for you: serviceaccount-e00d9fnpxv8qevt3pj, mk8s-node-sa
First one to create a registry - share it with the group. No need for everyone to duplicate the work.

IMPORTANT: You've done this before - please don't reach out for basic connection troubles