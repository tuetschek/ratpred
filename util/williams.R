
# Copyright 2015 Yvette Graham 
# 
# This file is part of nlp-williams.
# 
# nlp-williams is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# nlp-williams is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with nlp-williams.  If not, see <http://www.gnu.org/licenses/>

args <- commandArgs(T)
library(psych)

m1.h <- as.numeric(args[1])
m2.h <- as.numeric(args[2])
m1.m2 <- as.numeric(args[3])
samp.size <- as.numeric(args[4])


if( m1.h <= m2.h ){
    cat(paste( "error: r12 is required to be greater than r13 \n"))
    cat(paste("usage: R --no-save < williams.R r12 r13 r23 n","\n"))
    quit("no")
}

if(samp.size <= 0) {
    cat(paste( "error: bad sample size \n"))
    cat(paste( "usage: R --no-save < williams.R r12 r13 r23 n","\n"))
    quit("no")
}

p <- r.test( n = samp.size, 
            r12 = m1.h, 
            r13 = m2.h, 
            r23 = m1.m2, 
            twotailed = FALSE)$p

cat(paste( "-----------------------------------------" , "\n"))
cat(paste( "Williams Test for Increase in Correlation" , "\n"))
cat(paste( "\n"))
cat(paste( "    r12 correlation( human, metric_a ) : ", m1.h, "\n"))
cat(paste( "    r13 correlation( human, metric_b ) :" , m2.h, "\n"))
cat(paste( "    r23 correlation( metric_a, metric_b) :" , m1.m2, "\n"))
cat(paste( "\n"))
cat(paste( "    Sample size:" ,samp.size, "\n"))
cat(paste( "\n"))
cat(paste( "\n"))
cat(paste( "P-value:" , p, "\n"))
cat(paste( "-----------------------------------------" , "\n"))


