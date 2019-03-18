for a in `cat encodings.list`; do
  printf "encoding: $a  "
  iconv -f $a -t UTF-8 TrainData1.tsv /dev/null 2>&1 \
    && echo "ok: $a" || echo "fail: $a"
done | tee result.txt
