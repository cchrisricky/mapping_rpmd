for d in {000..099}
do
    cp MDrun.py "runs/$d"
    cp runjob.pbs "runs/$d"
    ( cd "runs/$d" && qsub runjob.pbs )
done
