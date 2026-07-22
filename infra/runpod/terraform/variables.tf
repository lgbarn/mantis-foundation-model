variable "data_center_id" {
  description = "Approved Secure Cloud datacenter that supports the S3-compatible volume API."
  type        = string
  default     = "US-MO-1"

  validation {
    condition     = var.data_center_id == "US-MO-1"
    error_message = "data_center_id must match the committed platform allowlist."
  }
}

variable "volume_name" {
  description = "Unique stable network-volume identity."
  type        = string
  default     = "mantis-v2-standard-volume-v1"

  validation {
    condition     = length(trimspace(var.volume_name)) > 0
    error_message = "volume_name must not be empty."
  }
}

variable "template_name" {
  description = "Unique private Pod-template identity."
  type        = string
  default     = "mantis-v2-private-template-v1"

  validation {
    condition     = length(trimspace(var.template_name)) > 0
    error_message = "template_name must not be empty."
  }
}

variable "image_digest" {
  description = "Public MantisV2 CUDA image pinned by sha256 digest."
  type        = string

  validation {
    condition     = can(regex("^[^[:space:]@]+@sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest must be an immutable registry reference ending in @sha256:<64 lowercase hex>."
  }
}
