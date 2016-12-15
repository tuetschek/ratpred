#!/usr/bin/env perl
#
# Given an experiment configuration file and several Makefile settings, this will
# create a short description

use strict;
use warnings;
use autodie;
use File::Basename;
use File::stat;
use File::Slurp;
use Getopt::Long;

my $USAGE = "Usage: ./$0 [-t TRAINING_SET] [-e] [-d] [-c CV] [-r] config.py \n";

my ( $eval_data, $training_set, $debug, $cv, $rands, $portion ) = ( 0, '', '', 0, '', 0, 1.0 );
GetOptions(
    'training_set|training|t=s' => \$training_set,
    'debug|d'                   => \$debug,
    'cv_runs|cv|c=s'            => \$cv,
    'rands|r'                   => \$rands,
    'train_portion|portion|p=f' => \$portion,
    'eval_data|eval|e'          => \$eval_data,
) or die($USAGE);
die($USAGE) if ( !@ARGV );

# Gather the settings from the command arguments and config files
my ( $data_set, $iters, $training_data, $gadgets, $run_setting, $nn_shape ) = ( '', '', '', '', '', '' );
my $config_data = read_file( $ARGV[0] );

# remove commented-out lines
$config_data =~ s/^\s*#.*$//gm;

# data set (devel -- default, eval -- mark)
if ($eval_data){
    $data_set = "\e[1;31mE\e[0m ";
}

# iterations
$iters = ( $config_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0];
$iters .= '/' . ( $config_data =~ /'batch_size'\s*:\s*([0-9]+)\s*,/ )[0];
$iters .= '/' . ( $config_data =~ /'alpha'\s*:\s*([.0-9eE-]+)\s*,/ )[0];
if ( $config_data =~ /'alpha_decay'\s*:\s*([.0-9eE-]+)\s*,/ and $1 > 0){
    $iters .= '^' . $1;
}
$iters =~ s/\/\//\/~\//;
$iters =~ s/\/$/\/~/;

if ($config_data =~ /'validation_size'\s*:\s*([0-9]+)\s*,/ and $1 != 0 ){
    $iters = ( ( $config_data =~ /'min_passes'\s*:\s*([0-9]+)\s*,/ )[0] // 1 ) . '-' . $iters;
    $iters .= ' V' . ( $config_data =~ /'validation_size'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '@' . ( ( $config_data =~ /'validation_freq'\s*:\s*([0-9]+)\s*,/ )[0] // 10);
    $iters .= ' I' . ( ( $config_data =~ /'improve_interval'\s*:\s*([0-9]+)\s*,/ )[0] // 10);
    $iters .= '@' . ( ( $config_data =~ /'top_k'\s*:\s*([0-9]+)\s*,/ )[0] // 5);
    if ( ( $config_data =~ /'bleu_validation_weight'\s*:\s*([01]\.?[0-9]*)/ ) ){
        my $val = $1;
        if ($val > 0.0){
            $iters .= ' B' . sprintf( "%.2g", $val );
        }
    }
}

# data style
$training_data = $training_set . ' -> ' . ( ( $config_data =~ /'target_col'\s*:\s*'([^']*)'/ )[0] // 'quality' );

# gadgets
$nn_shape .= ' E' . ( ( $config_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 50 );
$nn_shape .= '-N' . ( ( $config_data =~ /'num_hidden_units'\s*:\s*([0-9]*)/ )[0] // 128 );
$nn_shape .= '-T' . ( ( $config_data =~ /'tanh_layers'\s*:\s*([0-9]*)/ )[0] // 0 );

if ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ ){
    $nn_shape .= '-D' . ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ )[0];
}
$nn_shape .= ' ' . ( ( $config_data =~ /'cell_type'\s*:\s*'([^']*)'/ )[0] // 'lstm' );
$nn_shape .= ' +reuse'  if ( $config_data =~ /'reuse_embeddings'\s*:\s*True/ );
$nn_shape .= ' +ints'  if ( $config_data =~ /'predict_ints'\s*:\s*True/ );
$nn_shape .= ' +adgr'  if ( $config_data =~ /'optimizer_type'\s*:\s*'adagrad'/ );


if ($cv) {
    my @cv_runs = split /\s+/, $cv;
    $run_setting .= ' ' . scalar(@cv_runs) . 'CV';
}
if ($debug) {
    $run_setting .= ' DEBUG';
}
if ($rands) {
    $run_setting .= ' RANDS';
}
$run_setting =~ s/^ //;
$run_setting =~ s/ +/,/g;

# Print the output.
print "$training_data $data_set$iters$gadgets$nn_shape ($run_setting)";
