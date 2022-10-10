<p align = "center" draggable=”false”
   ><img src="https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png"
     width="200px"
     height="auto"/>
</p>



# <h1 align="center" id="heading">Week 4 - Deploying Containerized Applications using Docker Compose</h1>

## 📚 Learning Objectives

By the end of this session, you will be able to:

- Write a Docker file
- Write a Docker Compose File
- Route FastAPI to our Docker applications
- Deploy multiple Docker Containers locally and on AWS EC2

## 📦 Deliverables
- A screenshot of `docker container ls` command on AWS EC2
- A screenshot of http://ec2.ip.address:8000/docs

We are going to deploy a pretrained image segmentation model:

<https://github.com/tensorflow/models/tree/master/research/deeplab>

It is important that you research how to write a dockerfile and a docker-compose file.

#### Dockerfile
Please see [Docker](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/), [the Dockerfile cheatsheet](https://kapeli.com/cheat_sheets/Dockerfile.docset/Contents/Resources/Documents/index), and the [the Docker file guide](https://github.com/FourthBrain/MLO-4/blob/main/guides/dockerfile_guide.md).

#### Docker-Compose
Please see [Docker](https://docs.docker.com/compose/gettingstarted/), [FastAPI in Containers](https://fastapi.tiangolo.com/deployment/docker/), and [this](https://medium.com/swlh/python-with-docker-compose-fastapi-part-2-88e164d6ef86) and [this sample](https://github.com/paurakhsharma/python-microservice-fastapi/blob/master/docker-compose.yml).


## Create EC2 Instance

- Go to EC2 console: <https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1>#
- Create EC2 instance
- Pick amazon linux
- Pick instance type: At least t3.medium
- Create key-pair
- Download key
- Edit network
- Enable IPV4 address
- Open ports 8000-8002 from anywhere
- Launch Instance

## Install Dependencies

- Get the ip address of the instance
- Change key permissions to 400 (`chmod 400 key.pem`)
- SSH into the machine `ssh -i key.pem ec2-user@ec2.ip.address`
- Install git if needed (`sudo apt install git` for ubuntu based distros, `sudo yum install git` for amazon linux)
- Install Docker (`sudo apt install docker` for ubuntu based distros, `sudo yum install docker` for amazon linux)
- Start Docker (`sudo systemctl start docker`)
- Add user to docker group (`sudo usermod -aG docker ${USER}`)
- Logout and Login again through SSH to take the group changes into account
- Check if docker installed correctly (`docker run hello-world`)
- Install Docker-Compose

```
sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version
```

- Clone the repo (`git clone ...`)
- If there's permission issues with gitlab, generate ssh keys (`ssh-keygen`) and add them to github account
- CD into the folder (`cd cloned-repo`)

## Docker Compose

- Run all the endpoints (`docker-compose -f docker-compose.yaml up --build`)
- Create a request with docs (<http://ec2.ip.address:8000/docs>)
