# Stable RunPod resources

This root module owns exactly one 150 GB Standard network volume and one private
Pod template. The template uses a digest-pinned public image, exposes SSH only,
sets `/workspace/mantis` as the network-volume mount, and allocates no local
volume for data. Pod creation and termination belong to the lifecycle adapter.

OpenTofu is pinned to 1.11.4 and `Mastercard/restapi` is pinned to 3.0.0 because
the official RunPod provider does not expose network-volume or template
resources. The committed lock file contains registry checksums. During the
recorded `tofu init`, the registry supplied no GPG key, so OpenTofu explicitly
skipped provider signature validation; the provider is checksum-locked but is
not claimed to be signature-verified.

The provisioning key enters only through the ephemeral, sensitive OpenTofu
input. Export `RUNPOD_API_KEY` locally; the `just` recipes map it to
`TF_VAR_runpod_api_key` in the child environment. Never put the value in HCL,
`.tfvars`, state, a saved plan, a command argument, or a committed file.

## Read-only workflow

Formatting and validation do not contact RunPod:

```bash
just runpod-terraform-fmt
just runpod-terraform-validate
```

Copy `terraform.tfvars.example` to ignored `terraform.tfvars`, replace the image
example with the published digest, and make the same non-secret changes in an
ignored desired-contract JSON. Capture current volume/template inventory in the
strict shape shown by `inventory-empty.example.json`. Use
`state-addresses-empty.example.json` only for genuinely new local state.

Run adoption before planning:

```bash
just runpod-terraform-adoption \
  infra/runpod/terraform/desired-resources.json \
  infra/runpod/terraform/inventory.json \
  infra/runpod/terraform/state-addresses.json \
  infra/runpod/terraform/adoption.json
```

An absent identity may be created. One exact unmanaged match returns
`import_required` and an exact import ID; drift, duplicate names, or a state-ID
conflict fail closed. Import is a deliberate human action:

```bash
just runpod-terraform-import-human \
  restapi_object.network_volume /networkvolumes/VOLUME_ID
just runpod-terraform-import-human \
  restapi_object.pod_template /templates/TEMPLATE_ID
```

After recapturing inventory/state, create and inspect the plan. The `vars` and
saved-plan arguments are relative to this module; other paths are
repository-relative. The recipe reruns adoption before OpenTofu can propose a
change, renders plan JSON, and writes the policy report:

```bash
just runpod-terraform-plan \
  terraform.tfvars \
  infra/runpod/terraform/desired-resources.json \
  infra/runpod/terraform/inventory.json \
  infra/runpod/terraform/state-addresses.json \
  operator.tfplan \
  infra/runpod/terraform/operator.tfplan.json \
  infra/runpod/terraform/adoption.json \
  infra/runpod/terraform/policy.json
```

Policy permits only the two stable addresses with create, update, or no-op
actions. It rejects Pods, destroy/replacement, unknown policy fields, duplicate
addresses, secret values, provider/image drift, unsafe template fields, and any
adoption report that is not planable. An exact imported configuration must
produce two `no-op` actions.

## State and apply

`.terraform/`, state, saved plans and plan JSON, backend configuration,
generated variable files, inventory, adoption reports, policy reports, and
authorization files are local sensitive control-plane data and remain ignored.
No remote backend is configured.

Apply is never AFK. A human creates a short-lived authorization naming the two
stable addresses and matching the policy's saved-plan SHA-256 and desired
digest. The command re-hashes the opaque saved plan and refuses expired,
non-human, changed, or mismatched input:

```bash
just runpod-terraform-apply-human \
  operator.tfplan \
  infra/runpod/terraform/policy.json \
  infra/runpod/terraform/apply-authorization.json \
  2026-07-21T20:05:00Z \
  infra/runpod/terraform/apply-decision.json
```

This command can create billable storage. It requires fresh explicit human
authorization and must not be invoked by an unattended agent.
