terraform {
  required_version = "= 1.11.4"

  required_providers {
    restapi = {
      source  = "Mastercard/restapi"
      version = "= 3.0.0"
    }
  }
}
