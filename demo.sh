#!/bin/bash
set -e
echo What is the location of your conda environnement?
read pyconda
pyconda+='/bin/python'
echo "pyconda=${pyconda}"s
# dataset
echo "run pyconda manage.py dataset visualise"
(sleep 20s; echo '\r') | $pyconda manage.py dataset visualise
echo ''
echo "run pyconda manage.py dataset visualise --log"
(sleep 20s; echo '\r') | $pyconda manage.py dataset visualise --log
echo ''

# landscape
echo "run pyconda manage.py landscape visualise -w_size=50 --end_date='2000-01-10'"
(sleep 20s; echo '\r') | $pyconda manage.py landscape visualise -w_size=50 --end_date='2000-01-10'
echo ''
echo "run pyconda manage.py landscape visualise -w_size=50 --end_date='2000-03-10'"
(sleep 20s; echo '\r') | $pyconda manage.py landscape visualise -w_size=50 --end_date='2000-03-10'
echo ''
echo "run pyconda manage.py landscape visualise -w_size=50 --end_date='2008-07-15'"
(sleep 20s; echo '\r') | $pyconda manage.py landscape visualise -w_size=50 --end_date='2008-07-15'
echo ''
echo "run pyconda manage.py landscape visualise -w_size=50 --end_date='2008-09-15'"
(sleep 20s; echo '\r') | $pyconda manage.py landscape visualise -w_size=50 --end_date='2008-09-15'
echo ''

# norm
echo "run pyconda manage.py norm get -w_size=50"
$pyconda manage.py norm get -w_size=50
echo ''
echo "run pyconda manage.py norm visualise -w_size=50 --start_date='1997-01-03' --end_date='2000-03-10'"
(sleep 30s; echo '\r') | $pyconda manage.py norm visualise -w_size=50 --start_date='1997-01-03' --end_date='2000-03-10'
echo ''
echo "run pyconda manage.py norm crash_stats -w_size=50 -year=2000 --test --plot"
(sleep 60s; echo '\r') | $pyconda manage.py norm crash_stats -w_size=50 -year=2000 --test --plot
echo ''
echo "run pyconda manage.py norm visualise -w_size=50 --start_date='2005-06-08' --end_date='2008-09-15'"
(sleep 30s; echo '\r') | $pyconda manage.py norm visualise -w_size=50 --start_date='2005-06-08' --end_date='2008-09-15'
echo ''
echo "run pyconda manage.py norm crash_stats -w_size=50 -year=2008 --test --plot"
(sleep 60s; echo '\r') | $pyconda manage.py norm crash_stats -w_size=50 -year=2008 --test --plot
echo ''

# bottleneck
echo "run pyconda manage.py bottleneck get -w_size=50"
$pyconda manage.py bottleneck get -w_size=50
echo ''
echo "run pyconda manage.py bottleneck visualise -w_size=50 --start_date='1997-01-03' --end_date='2000-03-10'"
(sleep 30s; echo '\r') | $pyconda manage.py bottleneck visualise -w_size=50 --start_date='1997-01-03' --end_date='2000-03-10'
echo ''
echo "run pyconda manage.py bottleneck crash_stats -w_size=50 -year=2000 --test --plot"
(sleep 60s; echo '\r') | $pyconda manage.py bottleneck crash_stats -w_size=50 -year=2000 --test --plot
echo ''
echo "run pyconda manage.py bottleneck visualise -w_size=50 --start_date='2005-06-08' --end_date='2008-09-15'"
(sleep 30s; echo '\r') | $pyconda manage.py bottleneck visualise -w_size=50 --start_date='2005-06-08' --end_date='2008-09-15'
echo ''
echo "run pyconda manage.py bottleneck crash_stats -w_size=80 -year=2008 --test --plot"
(sleep 60s; echo '\r') | $pyconda manage.py bottleneck crash_stats -w_size=80 -year=2008 --test --plot
echo ''
