#!/bin/bash
#pyconda='/Users/vincentroy/anaconda3/envs/GMDA-TDAForFinancial/bin/python'
echo What is the location of your conda environnement?
read pyconda

# dataset
(sleep 20s; echo '\r') | $pyconda manage.py dataset visualise
echo ''
(sleep 20s; echo '\r') | $pyconda manage.py dataset visualise --log
echo ''

# landscape
(sleep 20s; echo '\r') | $pyconda manage.py landscape visualise -w_size=80 --end_date='2000-01-10'
echo ''

# norm
$pyconda manage.py norm get -w_size=50
echo ''
(sleep 30s; echo '\r') | $pyconda manage.py norm visualise -w_size=50 --start_date='1997-01-03' --end_date='2000-05-10'
echo ''
(sleep 60s; echo '\r') | $pyconda manage.py norm crash_stats -w_size=50 -year=2008 --test --plot
echo ''
(sleep 60s; echo '\r') | $pyconda manage.py norm crash_stats -w_size=50 -year=2000 --test --plot
echo ''

# bottleneck
$pyconda manage.py bottleneck get -w_size=50
echo ''
(sleep 30s; echo '\r') | $pyconda manage.py bottleneck visualise -w_size=80 --start_date='2005-01-03'
--end_date='2008-09-30'
echo ''
(sleep 60s; echo '\r') | $pyconda manage.py bottleneck crash_stats -w_size=80 -year=2008 --test --plot
echo ''
(sleep 60s; echo '\r') | $pyconda manage.py bottleneck crash_stats -w_size=80 -year=2000 --test --plot
echo ''
