#!/bin/bash

URLS=(
	"https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/thingsvision/bpe_simple_vocab_16e6.txt.gz"
	"https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/thingsvision/data/item_names.tsv"
	"https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/thingsvision/data/things_concepts.tsv"
)

FILES=(
	"bpe_simple_vocab_16e6.txt.gz"
	"item_names.tsv"
	"things_concepts.tsv"
)

v=$(python --version)
v=$(echo "$v" | cut -c10)
path="/Users/$(whoami)/anaconda3/lib/python3.$v/site-packages/thingsvision/"
subfolder="$(pwd)/data/"

mkdir "$subfolder"

for i in ${!URLS[@]}; do
	file=${FILES[i]}
	url=${URLS[i]}
	curl -O "$url"
	if [[ -f $file ]]; then
		echo "$url successfully downloaded"
		if [[ $i -eq 0 ]]; then
			mv "$file" "$path"
		else
			mv "$file" "$subfolder"
		fi
	else
		echo "$url not successfully downloaded"
		exit -1
	fi
done
cd ..
