variable "runpod_api_key" {
  description = "RunPod account provisioning key passed only through TF_VAR_runpod_api_key."
  type        = string
  sensitive   = true
  ephemeral   = true
}

provider "restapi" {
  uri                  = "https://rest.runpod.io/v1"
  bearer_token         = var.runpod_api_key
  id_attribute         = "id"
  create_method        = "POST"
  read_method          = "GET"
  update_method        = "PATCH"
  destroy_method       = "DELETE"
  write_returns_object = true
  debug                = false
}
