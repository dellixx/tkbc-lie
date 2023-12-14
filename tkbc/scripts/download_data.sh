
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

wget https://dl.fbaipublicfiles.com/tkbc/data.tar.gz
tar -xvzf data.tar.gz
rm data.tar.gz
