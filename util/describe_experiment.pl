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

# remove classification filter data so that they do not influence reading other settings
my $valid_crit_data = ( $config_data =~ /'validation_weights'\s*:\s*{([^}]*)}/s )[0];
$config_data =~ s/'validation_weights'\s*:\s*{[^}]*}//s;

# data set (devel -- default, eval -- mark)
if ($eval_data){
    $data_set = "\e[1;31mE\e[0m ";
}

# iterations
$iters = ( $config_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0];
if ( $config_data =~ /'pretrain_passes'\s*:\s*([0-9]+)\s*,/ and $1 > 0){
    $iters = $1 . '^'. $iters;
}
if ( $config_data =~ /'use_seq2seq'\s*:\s*True/ ){
    $iters = 'S' . (( $config_data =~ /'seq2seq_pretrain_passes'\s*:\s*([0-9]+)\s*,/ )[0] // 0 ) . ' ' . $iters;
}
if ((( $config_data =~ /'daclassif_pretrain_passes'\s*:\s*([0-9]+)\s*,/ )[0] // 0 ) > 0){
    $iters = 'D' . (( $config_data =~ /'daclassif_pretrain_passes'\s*:\s*([0-9]+)\s*,/ )[0] // 0 ) . ' ' . $iters;
}
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

    if ($valid_crit_data){
        $valid_crit_data =~ s/[\s\r\n:']//g;
        $valid_crit_data =~ s/([a-z])[a-z]*(?=[^a-z])/$1/g;
        $valid_crit_data =~ s/_//g;
        $valid_crit_data =~ s/,$//;
        $iters .= ' ' . $valid_crit_data;
    }
}

# data style
if ($portion < 1.0){
    $training_set .= '/' . $portion;
}
$training_set .= ' -slotn' if ( $config_data =~ /'delex_slot_names'\s*:\s*True/ );
$training_set .= ' -dlxda' if ( $config_data =~ /'delex_das'\s*:\s*True/ );
$training_set .= ' +lex' if ( $config_data !~ /'delex_slots'\s*:\s*'[^']/ );

my $target_col = 'Q';  # quality is the default
if ($config_data =~ /'target_col'\s*:\s*\[\s*([^\]]*)\]/){
    $target_col = $1;
    $target_col =~ s/['\s,]+/ /g;
    $target_col =~ s/\b([a-z])[a-z]*/$1/g;
    $target_col =~ s/ //g;
    $target_col = uc($target_col);
}
elsif ($config_data =~ /'target_col'\s*:\s*'([^']*)'/){
    $target_col = $1;
    $target_col = uc(substr($target_col, 0, 1));
}
$training_data = $training_set . ' -> ' . $target_col;


# gadgets
$nn_shape .= ' E' . ( ( $config_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 50 );
$nn_shape .= '-T' . ( ( $config_data =~ /'tanh_layers'\s*:\s*([0-9]*)/ )[0] // 0 );

if ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ ){
    $nn_shape .= '-D' . ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ )[0];
}
$nn_shape .= ' ' . ( ( $config_data =~ /'cell_type'\s*:\s*'([^']*)'/ )[0] // 'lstm' );
$nn_shape .= ' +bidi'  if ( $config_data =~ /'bidi'\s*:\s*True/ );
$nn_shape .= ' +w2v-s'  if ( $config_data =~ /'word2vec_embs'\s*:\s*'(?!trainable')/ );
$nn_shape .= ' +w2v-t'  if ( $config_data =~ /'word2vec_embs'\s*:\s*'trainable'/ );
$nn_shape .= ' +ce'  if ( $config_data =~ /'char_embs'\s*:\s*True/ );
$nn_shape .= ' +reuse'  if ( $config_data =~ /'reuse_embeddings'\s*:\s*True/ );
$nn_shape .= ' +da'  if ( $config_data =~ /'da_enc'\s*:\s*True/ );
$nn_shape .= ' -ref'  if ( $config_data =~ /'ref_enc'\s*:\s*False/ );
$nn_shape .= ' -hyp'  if ( $config_data =~ /'hyp_enc'\s*:\s*False/ );
$nn_shape .= ' +1/2s'  if ( $config_data =~ /'predict_halves'\s*:\s*True/ );
$nn_shape .= ' +co-t'  if ( $config_data =~ /'predict_coarse'\s*:\s*'train'/ );
$nn_shape .= ' +co-e'  if ( $config_data =~ /'predict_coarse'\s*:\s*'test'/ );
$nn_shape .= ' +ints'  if ( $config_data =~ /'predict_ints'\s*:\s*True/ );
$nn_shape .= ' +adgr'  if ( $config_data =~ /'optimizer_type'\s*:\s*'adagrad'/ );

if ($cv) {
    my @cv_runs = split /\s+/, $cv;
    if (@cv_runs > 1){
        $run_setting .= ' ' . scalar(@cv_runs) . 'CV';
    }
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
