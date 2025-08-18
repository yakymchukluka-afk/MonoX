terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name_prefix = "${var.project}-${var.environment}"
}

resource "aws_s3_bucket" "datasets" {
  bucket = "${local.name_prefix}-datasets"
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name_prefix}-artifacts"
}

resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

output "datasets_bucket" {
  value = aws_s3_bucket.datasets.bucket
}

output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "api_ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

