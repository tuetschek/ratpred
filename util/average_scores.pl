#!/usr/bin/env perl
#
# Averaging scores from different cross-validation runs.
#
# Always uses the last value found in the file, ie. the value from the last training iteration
# (if run against training log files).
#

use strict;
use warnings;
use autodie;
use File::Basename;
use File::stat;

my @patterns = ( 'Distance', 'Accuracy', 'Pearson', 'Spearman' );
my %data;
my %lines;

die("Usage: ./$0 file1.log file2.log [...]\n") if ( !@ARGV );

# filter ARGV to obtain just one file in each subdirectory
# TODO make this an option
my %files_by_dir;
foreach my $file (@ARGV) {
    my $dir = dirname($file);
    if ( !defined $files_by_dir{$dir} or ( ( stat( $files_by_dir{$dir} ) )->[9] < ( stat($file) )->[9] ) ) {
        $files_by_dir{$dir} = $file;
    }
}

# collect data
foreach my $file ( values %files_by_dir ) {

    open( my $fh, '<:utf8', $file );
    my %cur_data = ();

    while ( my $line = <$fh> ) {
        chomp $line;
        foreach my $pattern (@patterns) {
            if ( $line =~ /$pattern/ ) {

                if ( !$lines{$pattern} ) {
                    $lines{$pattern} = $line;
                }

                # extract all numbers on the line
                my @nums = $line =~ m/[0-9]+\.[0-9]+/g;

                # store them, keep only the last ones for this file
                $cur_data{$pattern} = \@nums;
            }
        }
    }

    # add to global list
    while ( my ( $p, $n ) = each %cur_data ) {
        if ( !$data{$p} ) {
            $data{$p} = [];
        }
        push @{$data{$p}}, $n;
    }
    close($fh);
}

# check if we have complete data (all logfiles must contain all defined patterns, at least BEST
# must be defined at all times)
foreach my $pattern (@patterns){
    if ( defined $data{$pattern} and ( scalar( @{ $data{$pattern} } ) != scalar( keys %files_by_dir ) ) ) {
        die("\e[1;31mThe scores are incomplete.\e[0m\n");
    }
}

# compute the averages
foreach my $pattern (@patterns) {

    my $values = $data{$pattern};
    next if ( !$values );

    # average data
    my $cumul = shift @$values;
    my $ctr   = 1;
    foreach my $valline (@$values) {
        for ( my $i = 0; $i < @$valline; ++$i ) {
            $cumul->[$i] += $valline->[$i];
        }
        $ctr++;
    }
    for ( my $i = 0; $i < @$cumul; ++$i ) {
        $cumul->[$i] /= $ctr;
    }

    # print out the result
    my $line = $lines{$pattern};
    my $out  = "";
    my $endpos = pos $line;

    while ( $line =~ m/([0-9]+\.[0-9]+)/g ) {
        my $num    = $1;
        $endpos = pos $line;
        $out .= substr( $line, length($out), $endpos - length($out) - length($num) );
        $out .= sprintf( "%.3f", shift @$cumul );
    }
    $out .= substr( $line, $endpos );
    print $out . "\n";
}

