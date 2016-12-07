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

my $USAGE = "Usage: ./$0 [--bleu-range=40:60] [--nist-range=3:6] file1.log file2.log [...]\n";

my ( $bleu_range, $nist_range ) = ( '40:80', '3:6' );
GetOptions(
    'bleu-range|bleu|b=s' => \$bleu_range,
    'nist-range|nist|n=s' => \$nist_range,
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

    if ( $line =~ /(Distance:)/i ) {
        $line =~ s/.*avg:/D/i;
        $line =~ s/\)$//;
        $pr = rg( 0, 5 , $line, 1 ) . "D $p\e[0m";
    }
    if ( $line =~ /(Accuracy:)/i ) {
        $line =~ s/.*Accuracy:/A/i;
        $line =~ s/%.*//;
        $pr .= rg( 0, 5 , $line, 1 ) . "A $p\e[0m";
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

