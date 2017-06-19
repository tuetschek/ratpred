#!/usr/bin/env perl
#
# Grep important scores from log files, print them out in a condensed format,
# with bash color codes.

use strict;
use warnings;
use autodie;
use File::Basename;
use File::stat;
use Getopt::Long;

my $USAGE = "Usage: ./$0 [--dist-range=5] file1.log file2.log [...]\n";

my ( $dist_range ) = ( 5 );
GetOptions(
    'dist-range|dist|d=i' => \$dist_range,
) or die($USAGE);
die($USAGE) if ( !@ARGV );


# Filter ARGV and get just the last log file
# TODO make this an option
my $file_to_process = undef;

foreach my $file (@ARGV) {
    next if ( !-e $file );
    if ( !defined $file_to_process or ( ( stat($file_to_process) )->[9] < ( stat($file) )->[9] ) ) {
        $file_to_process = $file;
    }
}
exit() if ( !defined $file_to_process );


# Process the file
open( my $fh, '<:utf8', $file_to_process );
my ( $pr, $lists, $bleu ) = ( '', '', '' );

while ( my $line = <$fh> ) {
    chomp $line;

    if ( $line =~ /(MAE:.*RMSE:)/i ) {
        my ($mae, $rmse) = ($line =~ /MAE: ([0-9.]+), RMSE: ([0-9.]+)/);
        $pr = rg( 0, $dist_range , $mae, 1 ) . "M $mae\e[0m  ";
        $pr .= rg( 0, $dist_range , $rmse, 1 ) . "R $rmse\e[0m  ";
    }
    if ( $line =~ /(Pearson|Spearman) correlation:/i ) {
        my $letter = substr($1, 0, 1);
        my ($sign, $corr, $pv) = ($line =~ /([ -])([0-9.]+) \(p-value ([0-9.]+)\)/);
        $pr .= rg( 0, 1, $corr ) . "$letter$sign$corr\e[0m  ";
    }

}

close($fh);

# Print the output
print "$pr\e[0m";

#
# Subs
#

# Get the bash 256 colors number given RGB (with values in the range 0-6)
sub rgb_code {
    my ( $r, $g, $b ) = @_;
    return "\e[38;5;" . ( ( 16 + ( 36 * $r ) + ( 6 * $g ) + $b ) ) . "m";
}

# Return red-green gradient rgb code
sub rg {
    my ( $t, $b, $v, $swap ) = @_;
    my $r = int( 0 + ( ( $v - $b ) / ( $t - $b ) * 6 ) );
    my $g = int( 6 - ( ( $v - $b ) / ( $t - $b ) * 6 ) );
    $r = 5 if ( $r > 5 );
    $r = 0 if ( $r < 0 );
    $g = 5 if ( $g > 5 );
    $g = 0 if ( $g < 0 );
    if ($swap){
        ($g, $r) = ($r, $g);
    }
    return rgb_code( $r, $g, 0 );
}

