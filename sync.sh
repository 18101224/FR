# Exclude patterns, including those from .gitignore and project-specific paths
EXCLUDE_OPTS="--exclude=checkpoint/ --exclude=results/ --exclude=gits/ --exclude=logs/ \
--exclude=run_v1/ --exclude=__pycache__/"

SOURCE_DIR="."

SERVERS=(
    "server8:/home/gpuadmin/mj/FaceRecognition/"
    "mine1:/home/mj/FaceRecognition/"
    
)

SERVERS=(
    "mine1:/home/mj/FaceRecognition/"
    "server8:/home/gpuadmin/mj/FaceRecognition/"
)
# Exclude checkpoint directory from syncing


for SERVER in "${SERVERS[@]}"; do

    rsync -avz --delete --checksum $EXCLUDE_OPTS "$SOURCE_DIR" "$SERVER" &
    echo "synchroized to $SERVER"

done 
wait
exit 0
# wait
# echo "****************************************"
# echo "*                                      *"
# echo "*   🚀🚀 SYNCHRONIZATION DONE 🚀🚀     *"
# echo "*                                      *"
# echo "****************************************"
# echo ""

