locals {
  lifecycle_owner = "terraform-stable-v1"
  volume = {
    dataCenterId = var.data_center_id
    name         = var.volume_name
    size         = 150
  }
  template = {
    category          = "NVIDIA"
    containerDiskInGb = 50
    dockerEntrypoint  = []
    dockerStartCmd    = []
    env               = {}
    imageName         = var.image_digest
    isPublic          = false
    isServerless      = false
    name              = var.template_name
    ports             = ["22/tcp"]
    readme            = "lifecycle-owner: ${local.lifecycle_owner}"
    volumeInGb        = 0
    volumeMountPath   = "/workspace/mantis"
  }
}

resource "restapi_object" "network_volume" {
  path                    = "/networkvolumes"
  data                    = jsonencode(local.volume)
  id_attribute            = "id"
  ignore_server_additions = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "restapi_object" "pod_template" {
  path                    = "/templates"
  data                    = jsonencode(local.template)
  id_attribute            = "id"
  ignore_server_additions = true

  lifecycle {
    prevent_destroy = true
  }
}
