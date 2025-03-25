mkdir results/segment-count
for i in {2..16}
do
   python3 source/main-refactored.py segment-count $i
done
