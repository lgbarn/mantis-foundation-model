output "network_volume_id" {
  description = "Stable RunPod network-volume ID for the Pod lifecycle adapter."
  value       = restapi_object.network_volume.id
}

output "pod_template_id" {
  description = "Private RunPod template ID for the Pod lifecycle adapter."
  value       = restapi_object.pod_template.id
}

output "desired_contract" {
  description = "Non-secret stable-resource contract inspected by the plan policy."
  value = {
    schema_version  = 1
    lifecycle_owner = local.lifecycle_owner
    volume          = local.volume
    template        = local.template
  }
}
