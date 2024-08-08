# Terraform configuration
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "self_driving_car" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "Self-Driving-Car"
  }
}
