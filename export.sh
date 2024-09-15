#!/bin/bash

./build.sh

# docker save datacentric_baseline | gzip -c > datacentric_baseline.tar.gz
dockername='dc_upload1'
docker save $dockername | gzip -c > ${dockername}.tar.gz
