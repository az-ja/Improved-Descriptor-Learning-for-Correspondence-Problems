#!/bin/bash
###To run, please set the following paths
cd ~/integrate_and_interpolate/flow/build
CPM=~/Thesis/cpm_test/build/CPM
FLOW=~/integrate_and_interpolate/flow/build/optical_flow
SINTEL=~/sintel/training/final
KITTI=~/kitti_2015/data_scene_flow/training/image_2

DESCRIPTION="learned_descriptors"

FOLDERS="alley_1 alley_2 ambush_5
               bamboo_1
               bandage_1 bandage_2 cave_2 cave_4 market_2
        market_5 mountain_1 shaman_2 shaman_3  sleeping_1
               temple_2 ambush_7 bamboo_2 sleeping_2 temple_3"
FOLDERS1="ambush_2"
FOLDERS2="ambush_4"
FOLDERS3="ambush_6"
FOLDERS4="market_6"

for f in $FOLDERS;
do
	for i in $(seq -s " " 1 49);
	do
		PADDED=$(printf "%04d" "$i")
		NEXT=$(($i+1))
		PADDEDNEXT=$(printf "%04d" "$NEXT")
		IMAGE1=${SINTEL}/${f}/frame_${PADDED}.png
		IMAGE2=${SINTEL}/${f}/frame_${PADDEDNEXT}.png
		NAME=${f}_${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/sintel_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/sintel_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
done


for f in $FOLDERS1;
do
	for i in $(seq -s " " 1 20);
	do
		PADDED=$(printf "%04d" "$i")
		NEXT=$(($i+1))
		PADDEDNEXT=$(printf "%04d" "$NEXT")
		IMAGE1=${SINTEL}/${f}/frame_${PADDED}.png
		IMAGE2=${SINTEL}/${f}/frame_${PADDEDNEXT}.png
		NAME=${f}_${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/sintel_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/sintel_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
done

for f in $FOLDERS2;
do
	for i in $(seq -s " " 1 32);
	do
		PADDED=$(printf "%04d" "$i")
		NEXT=$(($i+1))
		PADDEDNEXT=$(printf "%04d" "$NEXT")
		IMAGE1=${SINTEL}/${f}/frame_${PADDED}.png
		IMAGE2=${SINTEL}/${f}/frame_${PADDEDNEXT}.png
		NAME=${f}_${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/sintel_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/sintel_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
done

for f in $FOLDERS3;
do
	for i in $(seq -s " " 1 19);
	do
		PADDED=$(printf "%04d" "$i")
		NEXT=$(($i+1))
		PADDEDNEXT=$(printf "%04d" "$NEXT")
		IMAGE1=${SINTEL}/${f}/frame_${PADDED}.png
		IMAGE2=${SINTEL}/${f}/frame_${PADDEDNEXT}.png
		NAME=${f}_${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/sintel_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/sintel_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
done

for f in $FOLDERS4;
do
	for i in $(seq -s " " 1 39);
	do
		PADDED=$(printf "%04d" "$i")
		NEXT=$(($i+1))
		PADDEDNEXT=$(printf "%04d" "$NEXT")
		IMAGE1=${SINTEL}/${f}/frame_${PADDED}.png
		IMAGE2=${SINTEL}/${f}/frame_${PADDEDNEXT}.png
		NAME=${f}_${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/sintel_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/sintel_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
done



for i in $(seq -s " " 0 199);
	do
		PADDED=$(printf "%06d_10" "$i")
		PADDEDNEXT=$(printf "%06d_11" "$i")
		IMAGE1=${KITTI}/${PADDED}.png
		IMAGE2=${KITTI}/${PADDEDNEXT}.png
		NAME=${PADDED}_${DESCRIPTION}
		TEXTPATH=/home/azin/Thesis/evaluate/kitti_matches_all/${NAME}.txt
		FLOWPATH=/home/azin/Thesis/evaluate/kitti_flow_all/${NAME}.flo
		echo " --- "
		echo "running EVALUATE with $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH"
		echo " "
		#sh $EVALUATE $IMAGE1 $IMAGE2 $TEXTPATH $FLOWPATH
		$CPM $IMAGE1 $IMAGE2 $TEXTPATH
		$FLOW $IMAGE1 $IMAGE2 -m $TEXTPATH -o $FLOWPATH
	done
