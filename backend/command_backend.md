For build docker image:\
     `docker build -t eye-tracker-backend .`

For run docker conatainer:\
     `docker run -it -p 5000:5000 -v ${PWD}/model:/app/model -v ${PWD}/photos:/app/photos eye-tracker-backend`

Run in different terminal to test backend.\
For find out container ID:\
     `docker ps` 

For access to that docker container:\
     `docker exec <container ID> bash`