ҵ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??	
~
conv1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1a/kernel
w
!conv1a/kernel/Read/ReadVariableOpReadVariableOpconv1a/kernel*&
_output_shapes
:*
dtype0
n
conv1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1a/bias
g
conv1a/bias/Read/ReadVariableOpReadVariableOpconv1a/bias*
_output_shapes
:*
dtype0
~
conv1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1b/kernel
w
!conv1b/kernel/Read/ReadVariableOpReadVariableOpconv1b/kernel*&
_output_shapes
:*
dtype0
n
conv1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1b/bias
g
conv1b/bias/Read/ReadVariableOpReadVariableOpconv1b/bias*
_output_shapes
:*
dtype0
~
conv2a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2a/kernel
w
!conv2a/kernel/Read/ReadVariableOpReadVariableOpconv2a/kernel*&
_output_shapes
: *
dtype0
n
conv2a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2a/bias
g
conv2a/bias/Read/ReadVariableOpReadVariableOpconv2a/bias*
_output_shapes
: *
dtype0
~
conv2b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2b/kernel
w
!conv2b/kernel/Read/ReadVariableOpReadVariableOpconv2b/kernel*&
_output_shapes
:  *
dtype0
n
conv2b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2b/bias
g
conv2b/bias/Read/ReadVariableOpReadVariableOpconv2b/bias*
_output_shapes
: *
dtype0
~
conv2c/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv2c/kernel
w
!conv2c/kernel/Read/ReadVariableOpReadVariableOpconv2c/kernel*&
_output_shapes
: @*
dtype0
n
conv2c/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2c/bias
g
conv2c/bias/Read/ReadVariableOpReadVariableOpconv2c/bias*
_output_shapes
:@*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	 *
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	?	 *
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
: *
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

: *
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?2
value?2B?2 B?2
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
R
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api
h

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
R
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
h

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
f
0
1
2
 3
)4
*5
36
47
=8
>9
O10
P11
U12
V13
 
f
0
1
2
 3
)4
*5
36
47
=8
>9
O10
P11
U12
V13
?
[layer_regularization_losses
\metrics
trainable_variables
regularization_losses
]non_trainable_variables

^layers
	variables
_layer_metrics
 
 
 
 
?
`layer_regularization_losses
ametrics
trainable_variables
regularization_losses
bnon_trainable_variables

clayers
	variables
dlayer_metrics
YW
VARIABLE_VALUEconv1a/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1a/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
elayer_regularization_losses
fmetrics
trainable_variables
regularization_losses
gnon_trainable_variables

hlayers
	variables
ilayer_metrics
YW
VARIABLE_VALUEconv1b/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1b/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
jlayer_regularization_losses
kmetrics
!trainable_variables
"regularization_losses
lnon_trainable_variables

mlayers
#	variables
nlayer_metrics
 
 
 
?
olayer_regularization_losses
pmetrics
%trainable_variables
&regularization_losses
qnon_trainable_variables

rlayers
'	variables
slayer_metrics
YW
VARIABLE_VALUEconv2a/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2a/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?
tlayer_regularization_losses
umetrics
+trainable_variables
,regularization_losses
vnon_trainable_variables

wlayers
-	variables
xlayer_metrics
 
 
 
?
ylayer_regularization_losses
zmetrics
/trainable_variables
0regularization_losses
{non_trainable_variables

|layers
1	variables
}layer_metrics
YW
VARIABLE_VALUEconv2b/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2b/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?
~layer_regularization_losses
metrics
5trainable_variables
6regularization_losses
?non_trainable_variables
?layers
7	variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?metrics
9trainable_variables
:regularization_losses
?non_trainable_variables
?layers
;	variables
?layer_metrics
YW
VARIABLE_VALUEconv2c/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2c/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
?
 ?layer_regularization_losses
?metrics
?trainable_variables
@regularization_losses
?non_trainable_variables
?layers
A	variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?metrics
Ctrainable_variables
Dregularization_losses
?non_trainable_variables
?layers
E	variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?metrics
Gtrainable_variables
Hregularization_losses
?non_trainable_variables
?layers
I	variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?metrics
Ktrainable_variables
Lregularization_losses
?non_trainable_variables
?layers
M	variables
?layer_metrics
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
?
 ?layer_regularization_losses
?metrics
Qtrainable_variables
Rregularization_losses
?non_trainable_variables
?layers
S	variables
?layer_metrics
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
?
 ?layer_regularization_losses
?metrics
Wtrainable_variables
Xregularization_losses
?non_trainable_variables
?layers
Y	variables
?layer_metrics
 
 
 
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1a/kernelconv1a/biasconv1b/kernelconv1b/biasconv2a/kernelconv2a/biasconv2b/kernelconv2b/biasconv2c/kernelconv2c/biasdense1/kerneldense1/biasdense2/kerneldense2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_583656
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1a/kernel/Read/ReadVariableOpconv1a/bias/Read/ReadVariableOp!conv1b/kernel/Read/ReadVariableOpconv1b/bias/Read/ReadVariableOp!conv2a/kernel/Read/ReadVariableOpconv2a/bias/Read/ReadVariableOp!conv2b/kernel/Read/ReadVariableOpconv2b/bias/Read/ReadVariableOp!conv2c/kernel/Read/ReadVariableOpconv2c/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_584226
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1a/kernelconv1a/biasconv1b/kernelconv1b/biasconv2a/kernelconv2a/biasconv2b/kernelconv2b/biasconv2c/kernelconv2c/biasdense1/kerneldense1/biasdense2/kerneldense2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_584278??
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_583276

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?'
?
__inference__traced_save_584226
file_prefix,
(savev2_conv1a_kernel_read_readvariableop*
&savev2_conv1a_bias_read_readvariableop,
(savev2_conv1b_kernel_read_readvariableop*
&savev2_conv1b_bias_read_readvariableop,
(savev2_conv2a_kernel_read_readvariableop*
&savev2_conv2a_bias_read_readvariableop,
(savev2_conv2b_kernel_read_readvariableop*
&savev2_conv2b_bias_read_readvariableop,
(savev2_conv2c_kernel_read_readvariableop*
&savev2_conv2c_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1a_kernel_read_readvariableop&savev2_conv1a_bias_read_readvariableop(savev2_conv1b_kernel_read_readvariableop&savev2_conv1b_bias_read_readvariableop(savev2_conv2a_kernel_read_readvariableop&savev2_conv2a_bias_read_readvariableop(savev2_conv2b_kernel_read_readvariableop&savev2_conv2b_bias_read_readvariableop(savev2_conv2c_kernel_read_readvariableop&savev2_conv2c_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : :  : : @:@:	?	 : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@:%!

_output_shapes
:	?	 : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?

?
B__inference_conv2c_layer_call_and_return_conditional_losses_584074

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
A__inference_model_layer_call_and_return_conditional_losses_583590

inputs
conv1a_583548
conv1a_583550
conv1b_583553
conv1b_583555
conv2a_583559
conv2a_583561
conv2b_583565
conv2b_583567
conv2c_583571
conv2c_583573
dense1_583579
dense1_583581
dense2_583584
dense2_583586
identity??conv1a/StatefulPartitionedCall?conv1b/StatefulPartitionedCall?conv2a/StatefulPartitionedCall?conv2b/StatefulPartitionedCall?conv2c/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
process/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5831072
process/PartitionedCall?
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_583548conv1a_583550*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1a_layer_call_and_return_conditional_losses_5831312 
conv1a/StatefulPartitionedCall?
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_583553conv1b_583555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1b_layer_call_and_return_conditional_losses_5831582 
conv1b/StatefulPartitionedCall?
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool1c_layer_call_and_return_conditional_losses_5830532
pool1c/PartitionedCall?
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_583559conv2a_583561*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2a_layer_call_and_return_conditional_losses_5831862 
conv2a/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv2a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832192
dropout/PartitionedCall?
conv2b/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2b_583565conv2b_583567*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2b_layer_call_and_return_conditional_losses_5832432 
conv2b/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall'conv2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832762
dropout_1/PartitionedCall?
conv2c/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2c_583571conv2c_583573*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2c_layer_call_and_return_conditional_losses_5833002 
conv2c/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833332
dropout_2/PartitionedCall?
pool2d/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool2d_layer_call_and_return_conditional_losses_5830652
pool2d/PartitionedCall?
flatten/PartitionedCallPartitionedCallpool2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5833532
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_583579dense1_583581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_5833722 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_583584dense2_583586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5833992 
dense2/StatefulPartitionedCall?
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_584063

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832762
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_584116

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?<
?
"__inference__traced_restore_584278
file_prefix"
assignvariableop_conv1a_kernel"
assignvariableop_1_conv1a_bias$
 assignvariableop_2_conv1b_kernel"
assignvariableop_3_conv1b_bias$
 assignvariableop_4_conv2a_kernel"
assignvariableop_5_conv2a_bias$
 assignvariableop_6_conv2b_kernel"
assignvariableop_7_conv2b_bias$
 assignvariableop_8_conv2c_kernel"
assignvariableop_9_conv2c_bias%
!assignvariableop_10_dense1_kernel#
assignvariableop_11_dense1_bias%
!assignvariableop_12_dense2_kernel#
assignvariableop_13_dense2_bias
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1a_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1a_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1b_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1b_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2a_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2a_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2b_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2b_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2c_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2c_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
|
'__inference_conv2a_layer_call_fn_583989

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2a_layer_call_and_return_conditional_losses_5831862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????-::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
B__inference_dense2_layer_call_and_return_conditional_losses_583399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
D
(__inference_process_layer_call_fn_583924

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5830912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_584001

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_conv1b_layer_call_and_return_conditional_losses_583960

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????*Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????*Z
 
_user_specified_nameinputs
?
^
B__inference_pool1c_layer_call_and_return_conditional_losses_583053

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?=
?
A__inference_model_layer_call_and_return_conditional_losses_583416
input_1
conv1a_583142
conv1a_583144
conv1b_583169
conv1b_583171
conv2a_583197
conv2a_583199
conv2b_583254
conv2b_583256
conv2c_583311
conv2c_583313
dense1_583383
dense1_583385
dense2_583410
dense2_583412
identity??conv1a/StatefulPartitionedCall?conv1b/StatefulPartitionedCall?conv2a/StatefulPartitionedCall?conv2b/StatefulPartitionedCall?conv2c/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
process/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5830912
process/PartitionedCall?
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_583142conv1a_583144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1a_layer_call_and_return_conditional_losses_5831312 
conv1a/StatefulPartitionedCall?
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_583169conv1b_583171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1b_layer_call_and_return_conditional_losses_5831582 
conv1b/StatefulPartitionedCall?
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool1c_layer_call_and_return_conditional_losses_5830532
pool1c/PartitionedCall?
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_583197conv2a_583199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2a_layer_call_and_return_conditional_losses_5831862 
conv2a/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832142!
dropout/StatefulPartitionedCall?
conv2b/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2b_583254conv2b_583256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2b_layer_call_and_return_conditional_losses_5832432 
conv2b/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832712#
!dropout_1/StatefulPartitionedCall?
conv2c/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2c_583311conv2c_583313*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2c_layer_call_and_return_conditional_losses_5833002 
conv2c/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'conv2c/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833282#
!dropout_2/StatefulPartitionedCall?
pool2d/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool2d_layer_call_and_return_conditional_losses_5830652
pool2d/PartitionedCall?
flatten/PartitionedCallPartitionedCallpool2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5833532
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_583383dense1_583385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_5833722 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_583410dense2_583412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5833992 
dense2/StatefulPartitionedCall?
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

?
B__inference_conv2b_layer_call_and_return_conditional_losses_584027

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_584100

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
C
'__inference_pool2d_layer_call_fn_583071

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool2d_layer_call_and_return_conditional_losses_5830652
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_1_layer_call_fn_584058

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_584053

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_584006

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_583219

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_583328

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_conv1a_layer_call_and_return_conditional_losses_583131

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????S?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????S?
 
_user_specified_nameinputs
?	
?
&__inference_model_layer_call_fn_583621
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5835902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
c
*__inference_dropout_2_layer_call_fn_584105

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_584095

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
D
(__inference_process_layer_call_fn_583929

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5831072
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_583333

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_conv1a_layer_call_and_return_conditional_losses_583940

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????S?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????S?
 
_user_specified_nameinputs
?8
?
A__inference_model_layer_call_and_return_conditional_losses_583462
input_1
conv1a_583420
conv1a_583422
conv1b_583425
conv1b_583427
conv2a_583431
conv2a_583433
conv2b_583437
conv2b_583439
conv2c_583443
conv2c_583445
dense1_583451
dense1_583453
dense2_583456
dense2_583458
identity??conv1a/StatefulPartitionedCall?conv1b/StatefulPartitionedCall?conv2a/StatefulPartitionedCall?conv2b/StatefulPartitionedCall?conv2c/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
process/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5831072
process/PartitionedCall?
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_583420conv1a_583422*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1a_layer_call_and_return_conditional_losses_5831312 
conv1a/StatefulPartitionedCall?
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_583425conv1b_583427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1b_layer_call_and_return_conditional_losses_5831582 
conv1b/StatefulPartitionedCall?
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool1c_layer_call_and_return_conditional_losses_5830532
pool1c/PartitionedCall?
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_583431conv2a_583433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2a_layer_call_and_return_conditional_losses_5831862 
conv2a/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv2a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832192
dropout/PartitionedCall?
conv2b/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2b_583437conv2b_583439*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2b_layer_call_and_return_conditional_losses_5832432 
conv2b/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall'conv2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832762
dropout_1/PartitionedCall?
conv2c/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2c_583443conv2c_583445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2c_layer_call_and_return_conditional_losses_5833002 
conv2c/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833332
dropout_2/PartitionedCall?
pool2d/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool2d_layer_call_and_return_conditional_losses_5830652
pool2d/PartitionedCall?
flatten/PartitionedCallPartitionedCallpool2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5833532
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_583451dense1_583453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_5833722 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_583456dense2_583458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5833992 
dense2/StatefulPartitionedCall?
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?[
?
A__inference_model_layer_call_and_return_conditional_losses_583821

inputs)
%conv1a_conv2d_readvariableop_resource*
&conv1a_biasadd_readvariableop_resource)
%conv1b_conv2d_readvariableop_resource*
&conv1b_biasadd_readvariableop_resource)
%conv2a_conv2d_readvariableop_resource*
&conv2a_biasadd_readvariableop_resource)
%conv2b_conv2d_readvariableop_resource*
&conv2b_biasadd_readvariableop_resource)
%conv2c_conv2d_readvariableop_resource*
&conv2c_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identity??conv1a/BiasAdd/ReadVariableOp?conv1a/Conv2D/ReadVariableOp?conv1b/BiasAdd/ReadVariableOp?conv1b/Conv2D/ReadVariableOp?conv2a/BiasAdd/ReadVariableOp?conv2a/Conv2D/ReadVariableOp?conv2b/BiasAdd/ReadVariableOp?conv2b/Conv2D/ReadVariableOp?conv2c/BiasAdd/ReadVariableOp?conv2c/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?
process/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice/stack?
process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
process/strided_slice/stack_1?
process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
process/strided_slice/stack_2?
process/strided_sliceStridedSliceinputs$process/strided_slice/stack:output:0&process/strided_slice/stack_1:output:0&process/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
process/strided_slice?
process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice_1/stack?
process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
process/strided_slice_1/stack_1?
process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
process/strided_slice_1/stack_2?
process/strided_slice_1StridedSliceinputs&process/strided_slice_1/stack:output:0(process/strided_slice_1/stack_1:output:0(process/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
process/strided_slice_1u
process/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
process/concat/axis?
process/concatConcatV2process/strided_slice:output:0 process/strided_slice_1:output:0process/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
process/concat{
process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
process/resize/size?
process/resize/ResizeBilinearResizeBilinearprocess/concat:output:0process/resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
process/resize/ResizeBilinear?
conv1a/Conv2D/ReadVariableOpReadVariableOp%conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1a/Conv2D/ReadVariableOp?
conv1a/Conv2DConv2D.process/resize/ResizeBilinear:resized_images:0$conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
conv1a/Conv2D?
conv1a/BiasAdd/ReadVariableOpReadVariableOp&conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1a/BiasAdd/ReadVariableOp?
conv1a/BiasAddBiasAddconv1a/Conv2D:output:0%conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
conv1a/BiasAddu
conv1a/ReluReluconv1a/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
conv1a/Relu?
conv1b/Conv2D/ReadVariableOpReadVariableOp%conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1b/Conv2D/ReadVariableOp?
conv1b/Conv2DConv2Dconv1a/Relu:activations:0$conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
conv1b/Conv2D?
conv1b/BiasAdd/ReadVariableOpReadVariableOp&conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1b/BiasAdd/ReadVariableOp?
conv1b/BiasAddBiasAddconv1b/Conv2D:output:0%conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
conv1b/BiasAddu
conv1b/ReluReluconv1b/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
conv1b/Relu?
pool1c/MaxPoolMaxPoolconv1b/Relu:activations:0*/
_output_shapes
:?????????-*
ksize
*
paddingSAME*
strides
2
pool1c/MaxPool?
conv2a/Conv2D/ReadVariableOpReadVariableOp%conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2a/Conv2D/ReadVariableOp?
conv2a/Conv2DConv2Dpool1c/MaxPool:output:0$conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2a/Conv2D?
conv2a/BiasAdd/ReadVariableOpReadVariableOp&conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2a/BiasAdd/ReadVariableOp?
conv2a/BiasAddBiasAddconv2a/Conv2D:output:0%conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2a/BiasAddu
conv2a/ReluReluconv2a/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2a/Relu?
dropout/IdentityIdentityconv2a/Relu:activations:0*
T0*/
_output_shapes
:????????? 2
dropout/Identity?
conv2b/Conv2D/ReadVariableOpReadVariableOp%conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2b/Conv2D/ReadVariableOp?
conv2b/Conv2DConv2Ddropout/Identity:output:0$conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2b/Conv2D?
conv2b/BiasAdd/ReadVariableOpReadVariableOp&conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2b/BiasAdd/ReadVariableOp?
conv2b/BiasAddBiasAddconv2b/Conv2D:output:0%conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2b/BiasAddu
conv2b/ReluReluconv2b/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2b/Relu?
dropout_1/IdentityIdentityconv2b/Relu:activations:0*
T0*/
_output_shapes
:????????? 2
dropout_1/Identity?
conv2c/Conv2D/ReadVariableOpReadVariableOp%conv2c_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2c/Conv2D/ReadVariableOp?
conv2c/Conv2DConv2Ddropout_1/Identity:output:0$conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2c/Conv2D?
conv2c/BiasAdd/ReadVariableOpReadVariableOp&conv2c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2c/BiasAdd/ReadVariableOp?
conv2c/BiasAddBiasAddconv2c/Conv2D:output:0%conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2c/BiasAddu
conv2c/ReluReluconv2c/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2c/Relu?
dropout_2/IdentityIdentityconv2c/Relu:activations:0*
T0*/
_output_shapes
:?????????@2
dropout_2/Identity?
pool2d/AvgPoolAvgPooldropout_2/Identity:output:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
pool2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapepool2d/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/BiasAddm
dense2/TanhTanhdense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense2/Tanh?
IdentityIdentitydense2/Tanh:y:0^conv1a/BiasAdd/ReadVariableOp^conv1a/Conv2D/ReadVariableOp^conv1b/BiasAdd/ReadVariableOp^conv1b/Conv2D/ReadVariableOp^conv2a/BiasAdd/ReadVariableOp^conv2a/Conv2D/ReadVariableOp^conv2b/BiasAdd/ReadVariableOp^conv2b/Conv2D/ReadVariableOp^conv2c/BiasAdd/ReadVariableOp^conv2c/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
conv1a/BiasAdd/ReadVariableOpconv1a/BiasAdd/ReadVariableOp2<
conv1a/Conv2D/ReadVariableOpconv1a/Conv2D/ReadVariableOp2>
conv1b/BiasAdd/ReadVariableOpconv1b/BiasAdd/ReadVariableOp2<
conv1b/Conv2D/ReadVariableOpconv1b/Conv2D/ReadVariableOp2>
conv2a/BiasAdd/ReadVariableOpconv2a/BiasAdd/ReadVariableOp2<
conv2a/Conv2D/ReadVariableOpconv2a/Conv2D/ReadVariableOp2>
conv2b/BiasAdd/ReadVariableOpconv2b/BiasAdd/ReadVariableOp2<
conv2b/Conv2D/ReadVariableOpconv2b/Conv2D/ReadVariableOp2>
conv2c/BiasAdd/ReadVariableOpconv2c/BiasAdd/ReadVariableOp2<
conv2c/Conv2D/ReadVariableOpconv2c/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_process_layer_call_and_return_conditional_losses_583919

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2strided_slice:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
resize/size?
resize/ResizeBilinearResizeBilinearconcat:output:0resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_584110

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?f
?

!__inference__wrapped_model_583047
input_1/
+model_conv1a_conv2d_readvariableop_resource0
,model_conv1a_biasadd_readvariableop_resource/
+model_conv1b_conv2d_readvariableop_resource0
,model_conv1b_biasadd_readvariableop_resource/
+model_conv2a_conv2d_readvariableop_resource0
,model_conv2a_biasadd_readvariableop_resource/
+model_conv2b_conv2d_readvariableop_resource0
,model_conv2b_biasadd_readvariableop_resource/
+model_conv2c_conv2d_readvariableop_resource0
,model_conv2c_biasadd_readvariableop_resource/
+model_dense1_matmul_readvariableop_resource0
,model_dense1_biasadd_readvariableop_resource/
+model_dense2_matmul_readvariableop_resource0
,model_dense2_biasadd_readvariableop_resource
identity??#model/conv1a/BiasAdd/ReadVariableOp?"model/conv1a/Conv2D/ReadVariableOp?#model/conv1b/BiasAdd/ReadVariableOp?"model/conv1b/Conv2D/ReadVariableOp?#model/conv2a/BiasAdd/ReadVariableOp?"model/conv2a/Conv2D/ReadVariableOp?#model/conv2b/BiasAdd/ReadVariableOp?"model/conv2b/Conv2D/ReadVariableOp?#model/conv2c/BiasAdd/ReadVariableOp?"model/conv2c/Conv2D/ReadVariableOp?#model/dense1/BiasAdd/ReadVariableOp?"model/dense1/MatMul/ReadVariableOp?#model/dense2/BiasAdd/ReadVariableOp?"model/dense2/MatMul/ReadVariableOp?
!model/process/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!model/process/strided_slice/stack?
#model/process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#model/process/strided_slice/stack_1?
#model/process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#model/process/strided_slice/stack_2?
model/process/strided_sliceStridedSliceinput_1*model/process/strided_slice/stack:output:0,model/process/strided_slice/stack_1:output:0,model/process/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
model/process/strided_slice?
#model/process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2%
#model/process/strided_slice_1/stack?
%model/process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%model/process/strided_slice_1/stack_1?
%model/process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%model/process/strided_slice_1/stack_2?
model/process/strided_slice_1StridedSliceinput_1,model/process/strided_slice_1/stack:output:0.model/process/strided_slice_1/stack_1:output:0.model/process/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
model/process/strided_slice_1?
model/process/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
model/process/concat/axis?
model/process/concatConcatV2$model/process/strided_slice:output:0&model/process/strided_slice_1:output:0"model/process/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
model/process/concat?
model/process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
model/process/resize/size?
#model/process/resize/ResizeBilinearResizeBilinearmodel/process/concat:output:0"model/process/resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2%
#model/process/resize/ResizeBilinear?
"model/conv1a/Conv2D/ReadVariableOpReadVariableOp+model_conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv1a/Conv2D/ReadVariableOp?
model/conv1a/Conv2DConv2D4model/process/resize/ResizeBilinear:resized_images:0*model/conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
model/conv1a/Conv2D?
#model/conv1a/BiasAdd/ReadVariableOpReadVariableOp,model_conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv1a/BiasAdd/ReadVariableOp?
model/conv1a/BiasAddBiasAddmodel/conv1a/Conv2D:output:0+model/conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
model/conv1a/BiasAdd?
model/conv1a/ReluRelumodel/conv1a/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
model/conv1a/Relu?
"model/conv1b/Conv2D/ReadVariableOpReadVariableOp+model_conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02$
"model/conv1b/Conv2D/ReadVariableOp?
model/conv1b/Conv2DConv2Dmodel/conv1a/Relu:activations:0*model/conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
model/conv1b/Conv2D?
#model/conv1b/BiasAdd/ReadVariableOpReadVariableOp,model_conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv1b/BiasAdd/ReadVariableOp?
model/conv1b/BiasAddBiasAddmodel/conv1b/Conv2D:output:0+model/conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
model/conv1b/BiasAdd?
model/conv1b/ReluRelumodel/conv1b/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
model/conv1b/Relu?
model/pool1c/MaxPoolMaxPoolmodel/conv1b/Relu:activations:0*/
_output_shapes
:?????????-*
ksize
*
paddingSAME*
strides
2
model/pool1c/MaxPool?
"model/conv2a/Conv2D/ReadVariableOpReadVariableOp+model_conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2a/Conv2D/ReadVariableOp?
model/conv2a/Conv2DConv2Dmodel/pool1c/MaxPool:output:0*model/conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model/conv2a/Conv2D?
#model/conv2a/BiasAdd/ReadVariableOpReadVariableOp,model_conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2a/BiasAdd/ReadVariableOp?
model/conv2a/BiasAddBiasAddmodel/conv2a/Conv2D:output:0+model/conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model/conv2a/BiasAdd?
model/conv2a/ReluRelumodel/conv2a/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model/conv2a/Relu?
model/dropout/IdentityIdentitymodel/conv2a/Relu:activations:0*
T0*/
_output_shapes
:????????? 2
model/dropout/Identity?
"model/conv2b/Conv2D/ReadVariableOpReadVariableOp+model_conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02$
"model/conv2b/Conv2D/ReadVariableOp?
model/conv2b/Conv2DConv2Dmodel/dropout/Identity:output:0*model/conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model/conv2b/Conv2D?
#model/conv2b/BiasAdd/ReadVariableOpReadVariableOp,model_conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2b/BiasAdd/ReadVariableOp?
model/conv2b/BiasAddBiasAddmodel/conv2b/Conv2D:output:0+model/conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model/conv2b/BiasAdd?
model/conv2b/ReluRelumodel/conv2b/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model/conv2b/Relu?
model/dropout_1/IdentityIdentitymodel/conv2b/Relu:activations:0*
T0*/
_output_shapes
:????????? 2
model/dropout_1/Identity?
"model/conv2c/Conv2D/ReadVariableOpReadVariableOp+model_conv2c_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02$
"model/conv2c/Conv2D/ReadVariableOp?
model/conv2c/Conv2DConv2D!model/dropout_1/Identity:output:0*model/conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2c/Conv2D?
#model/conv2c/BiasAdd/ReadVariableOpReadVariableOp,model_conv2c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2c/BiasAdd/ReadVariableOp?
model/conv2c/BiasAddBiasAddmodel/conv2c/Conv2D:output:0+model/conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2c/BiasAdd?
model/conv2c/ReluRelumodel/conv2c/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2c/Relu?
model/dropout_2/IdentityIdentitymodel/conv2c/Relu:activations:0*
T0*/
_output_shapes
:?????????@2
model/dropout_2/Identity?
model/pool2d/AvgPoolAvgPool!model/dropout_2/Identity:output:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
model/pool2d/AvgPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten/Const?
model/flatten/ReshapeReshapemodel/pool2d/AvgPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
model/flatten/Reshape?
"model/dense1/MatMul/ReadVariableOpReadVariableOp+model_dense1_matmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02$
"model/dense1/MatMul/ReadVariableOp?
model/dense1/MatMulMatMulmodel/flatten/Reshape:output:0*model/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense1/MatMul?
#model/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/dense1/BiasAdd/ReadVariableOp?
model/dense1/BiasAddBiasAddmodel/dense1/MatMul:product:0+model/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense1/BiasAdd
model/dense1/ReluRelumodel/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/dense1/Relu?
"model/dense2/MatMul/ReadVariableOpReadVariableOp+model_dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"model/dense2/MatMul/ReadVariableOp?
model/dense2/MatMulMatMulmodel/dense1/Relu:activations:0*model/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense2/MatMul?
#model/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/dense2/BiasAdd/ReadVariableOp?
model/dense2/BiasAddBiasAddmodel/dense2/MatMul:product:0+model/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense2/BiasAdd
model/dense2/TanhTanhmodel/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense2/Tanh?
IdentityIdentitymodel/dense2/Tanh:y:0$^model/conv1a/BiasAdd/ReadVariableOp#^model/conv1a/Conv2D/ReadVariableOp$^model/conv1b/BiasAdd/ReadVariableOp#^model/conv1b/Conv2D/ReadVariableOp$^model/conv2a/BiasAdd/ReadVariableOp#^model/conv2a/Conv2D/ReadVariableOp$^model/conv2b/BiasAdd/ReadVariableOp#^model/conv2b/Conv2D/ReadVariableOp$^model/conv2c/BiasAdd/ReadVariableOp#^model/conv2c/Conv2D/ReadVariableOp$^model/dense1/BiasAdd/ReadVariableOp#^model/dense1/MatMul/ReadVariableOp$^model/dense2/BiasAdd/ReadVariableOp#^model/dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2J
#model/conv1a/BiasAdd/ReadVariableOp#model/conv1a/BiasAdd/ReadVariableOp2H
"model/conv1a/Conv2D/ReadVariableOp"model/conv1a/Conv2D/ReadVariableOp2J
#model/conv1b/BiasAdd/ReadVariableOp#model/conv1b/BiasAdd/ReadVariableOp2H
"model/conv1b/Conv2D/ReadVariableOp"model/conv1b/Conv2D/ReadVariableOp2J
#model/conv2a/BiasAdd/ReadVariableOp#model/conv2a/BiasAdd/ReadVariableOp2H
"model/conv2a/Conv2D/ReadVariableOp"model/conv2a/Conv2D/ReadVariableOp2J
#model/conv2b/BiasAdd/ReadVariableOp#model/conv2b/BiasAdd/ReadVariableOp2H
"model/conv2b/Conv2D/ReadVariableOp"model/conv2b/Conv2D/ReadVariableOp2J
#model/conv2c/BiasAdd/ReadVariableOp#model/conv2c/BiasAdd/ReadVariableOp2H
"model/conv2c/Conv2D/ReadVariableOp"model/conv2c/Conv2D/ReadVariableOp2J
#model/dense1/BiasAdd/ReadVariableOp#model/dense1/BiasAdd/ReadVariableOp2H
"model/dense1/MatMul/ReadVariableOp"model/dense1/MatMul/ReadVariableOp2J
#model/dense2/BiasAdd/ReadVariableOp#model/dense2/BiasAdd/ReadVariableOp2H
"model/dense2/MatMul/ReadVariableOp"model/dense2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
_
C__inference_process_layer_call_and_return_conditional_losses_583091

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2strided_slice:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
resize/size?
resize/ResizeBilinearResizeBilinearconcat:output:0resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_conv1b_layer_call_fn_583969

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1b_layer_call_and_return_conditional_losses_5831582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????*Z::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????*Z
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_583271

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_583656
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_5830472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
a
(__inference_dropout_layer_call_fn_584011

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_conv1b_layer_call_and_return_conditional_losses_583158

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????*Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????*Z
 
_user_specified_nameinputs
?
|
'__inference_conv2b_layer_call_fn_584036

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2b_layer_call_and_return_conditional_losses_5832432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_conv2c_layer_call_fn_584083

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2c_layer_call_and_return_conditional_losses_5833002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
&__inference_model_layer_call_fn_583854

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5835112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
B__inference_dense1_layer_call_and_return_conditional_losses_583372

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
B__inference_dense1_layer_call_and_return_conditional_losses_584132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_583353

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_584121

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5833532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
&__inference_model_layer_call_fn_583542
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5835112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
|
'__inference_dense1_layer_call_fn_584141

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_5833722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
&__inference_model_layer_call_fn_583887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_5835902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_conv1a_layer_call_fn_583949

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1a_layer_call_and_return_conditional_losses_5831312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????*Z2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????S?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????S?
 
_user_specified_nameinputs
?=
?
A__inference_model_layer_call_and_return_conditional_losses_583511

inputs
conv1a_583469
conv1a_583471
conv1b_583474
conv1b_583476
conv2a_583480
conv2a_583482
conv2b_583486
conv2b_583488
conv2c_583492
conv2c_583494
dense1_583500
dense1_583502
dense2_583505
dense2_583507
identity??conv1a/StatefulPartitionedCall?conv1b/StatefulPartitionedCall?conv2a/StatefulPartitionedCall?conv2b/StatefulPartitionedCall?conv2c/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
process/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????S?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_process_layer_call_and_return_conditional_losses_5830912
process/PartitionedCall?
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_583469conv1a_583471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1a_layer_call_and_return_conditional_losses_5831312 
conv1a/StatefulPartitionedCall?
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_583474conv1b_583476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1b_layer_call_and_return_conditional_losses_5831582 
conv1b/StatefulPartitionedCall?
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????-* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool1c_layer_call_and_return_conditional_losses_5830532
pool1c/PartitionedCall?
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_583480conv2a_583482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2a_layer_call_and_return_conditional_losses_5831862 
conv2a/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832142!
dropout/StatefulPartitionedCall?
conv2b/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2b_583486conv2b_583488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2b_layer_call_and_return_conditional_losses_5832432 
conv2b/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_5832712#
!dropout_1/StatefulPartitionedCall?
conv2c/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2c_583492conv2c_583494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2c_layer_call_and_return_conditional_losses_5833002 
conv2c/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'conv2c/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_5833282#
!dropout_2/StatefulPartitionedCall?
pool2d/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool2d_layer_call_and_return_conditional_losses_5830652
pool2d/PartitionedCall?
flatten/PartitionedCallPartitionedCallpool2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5833532
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_583500dense1_583502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense1_layer_call_and_return_conditional_losses_5833722 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_583505dense2_583507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5833992 
dense2/StatefulPartitionedCall?
IdentityIdentity'dense2/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_583214

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_pool2d_layer_call_and_return_conditional_losses_583065

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_conv2b_layer_call_and_return_conditional_losses_583243

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_584016

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_5832192
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?w
?
A__inference_model_layer_call_and_return_conditional_losses_583749

inputs)
%conv1a_conv2d_readvariableop_resource*
&conv1a_biasadd_readvariableop_resource)
%conv1b_conv2d_readvariableop_resource*
&conv1b_biasadd_readvariableop_resource)
%conv2a_conv2d_readvariableop_resource*
&conv2a_biasadd_readvariableop_resource)
%conv2b_conv2d_readvariableop_resource*
&conv2b_biasadd_readvariableop_resource)
%conv2c_conv2d_readvariableop_resource*
&conv2c_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identity??conv1a/BiasAdd/ReadVariableOp?conv1a/Conv2D/ReadVariableOp?conv1b/BiasAdd/ReadVariableOp?conv1b/Conv2D/ReadVariableOp?conv2a/BiasAdd/ReadVariableOp?conv2a/Conv2D/ReadVariableOp?conv2b/BiasAdd/ReadVariableOp?conv2b/Conv2D/ReadVariableOp?conv2c/BiasAdd/ReadVariableOp?conv2c/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?
process/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice/stack?
process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
process/strided_slice/stack_1?
process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
process/strided_slice/stack_2?
process/strided_sliceStridedSliceinputs$process/strided_slice/stack:output:0&process/strided_slice/stack_1:output:0&process/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
process/strided_slice?
process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice_1/stack?
process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2!
process/strided_slice_1/stack_1?
process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
process/strided_slice_1/stack_2?
process/strided_slice_1StridedSliceinputs&process/strided_slice_1/stack:output:0(process/strided_slice_1/stack_1:output:0(process/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
process/strided_slice_1u
process/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
process/concat/axis?
process/concatConcatV2process/strided_slice:output:0 process/strided_slice_1:output:0process/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
process/concat{
process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
process/resize/size?
process/resize/ResizeBilinearResizeBilinearprocess/concat:output:0process/resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
process/resize/ResizeBilinear?
conv1a/Conv2D/ReadVariableOpReadVariableOp%conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1a/Conv2D/ReadVariableOp?
conv1a/Conv2DConv2D.process/resize/ResizeBilinear:resized_images:0$conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
conv1a/Conv2D?
conv1a/BiasAdd/ReadVariableOpReadVariableOp&conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1a/BiasAdd/ReadVariableOp?
conv1a/BiasAddBiasAddconv1a/Conv2D:output:0%conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
conv1a/BiasAddu
conv1a/ReluReluconv1a/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
conv1a/Relu?
conv1b/Conv2D/ReadVariableOpReadVariableOp%conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1b/Conv2D/ReadVariableOp?
conv1b/Conv2DConv2Dconv1a/Relu:activations:0$conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z*
paddingSAME*
strides
2
conv1b/Conv2D?
conv1b/BiasAdd/ReadVariableOpReadVariableOp&conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1b/BiasAdd/ReadVariableOp?
conv1b/BiasAddBiasAddconv1b/Conv2D:output:0%conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*Z2
conv1b/BiasAddu
conv1b/ReluReluconv1b/BiasAdd:output:0*
T0*/
_output_shapes
:?????????*Z2
conv1b/Relu?
pool1c/MaxPoolMaxPoolconv1b/Relu:activations:0*/
_output_shapes
:?????????-*
ksize
*
paddingSAME*
strides
2
pool1c/MaxPool?
conv2a/Conv2D/ReadVariableOpReadVariableOp%conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2a/Conv2D/ReadVariableOp?
conv2a/Conv2DConv2Dpool1c/MaxPool:output:0$conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2a/Conv2D?
conv2a/BiasAdd/ReadVariableOpReadVariableOp&conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2a/BiasAdd/ReadVariableOp?
conv2a/BiasAddBiasAddconv2a/Conv2D:output:0%conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2a/BiasAddu
conv2a/ReluReluconv2a/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2a/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulconv2a/Relu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/Mulw
dropout/dropout/ShapeShapeconv2a/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/Mul_1?
conv2b/Conv2D/ReadVariableOpReadVariableOp%conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2b/Conv2D/ReadVariableOp?
conv2b/Conv2DConv2Ddropout/dropout/Mul_1:z:0$conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2b/Conv2D?
conv2b/BiasAdd/ReadVariableOpReadVariableOp&conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2b/BiasAdd/ReadVariableOp?
conv2b/BiasAddBiasAddconv2b/Conv2D:output:0%conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2b/BiasAddu
conv2b/ReluReluconv2b/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2b/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulconv2b/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul{
dropout_1/dropout/ShapeShapeconv2b/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_1/dropout/Mul_1?
conv2c/Conv2D/ReadVariableOpReadVariableOp%conv2c_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv2c/Conv2D/ReadVariableOp?
conv2c/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0$conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2c/Conv2D?
conv2c/BiasAdd/ReadVariableOpReadVariableOp&conv2c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2c/BiasAdd/ReadVariableOp?
conv2c/BiasAddBiasAddconv2c/Conv2D:output:0%conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2c/BiasAddu
conv2c/ReluReluconv2c/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2c/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulconv2c/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul{
dropout_2/dropout/ShapeShapeconv2c/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_2/dropout/Mul_1?
pool2d/AvgPoolAvgPooldropout_2/dropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
pool2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapepool2d/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	?	 *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense2/BiasAddm
dense2/TanhTanhdense2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense2/Tanh?
IdentityIdentitydense2/Tanh:y:0^conv1a/BiasAdd/ReadVariableOp^conv1a/Conv2D/ReadVariableOp^conv1b/BiasAdd/ReadVariableOp^conv1b/Conv2D/ReadVariableOp^conv2a/BiasAdd/ReadVariableOp^conv2a/Conv2D/ReadVariableOp^conv2b/BiasAdd/ReadVariableOp^conv2b/Conv2D/ReadVariableOp^conv2c/BiasAdd/ReadVariableOp^conv2c/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
conv1a/BiasAdd/ReadVariableOpconv1a/BiasAdd/ReadVariableOp2<
conv1a/Conv2D/ReadVariableOpconv1a/Conv2D/ReadVariableOp2>
conv1b/BiasAdd/ReadVariableOpconv1b/BiasAdd/ReadVariableOp2<
conv1b/Conv2D/ReadVariableOpconv1b/Conv2D/ReadVariableOp2>
conv2a/BiasAdd/ReadVariableOpconv2a/BiasAdd/ReadVariableOp2<
conv2a/Conv2D/ReadVariableOpconv2a/Conv2D/ReadVariableOp2>
conv2b/BiasAdd/ReadVariableOpconv2b/BiasAdd/ReadVariableOp2<
conv2b/Conv2D/ReadVariableOpconv2b/Conv2D/ReadVariableOp2>
conv2c/BiasAdd/ReadVariableOpconv2c/BiasAdd/ReadVariableOp2<
conv2c/Conv2D/ReadVariableOpconv2c/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_process_layer_call_and_return_conditional_losses_583903

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2strided_slice:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
resize/size?
resize/ResizeBilinearResizeBilinearconcat:output:0resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2a_layer_call_and_return_conditional_losses_583980

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????-
 
_user_specified_nameinputs
?

?
B__inference_conv2c_layer_call_and_return_conditional_losses_583300

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_584048

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_dense2_layer_call_fn_584161

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense2_layer_call_and_return_conditional_losses_5833992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
_
C__inference_process_layer_call_and_return_conditional_losses_583107

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*
ellipsis_mask*
end_mask2
strided_slice_1e
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2strided_slice:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"S   ?   2
resize/size?
resize/ResizeBilinearResizeBilinearconcat:output:0resize/size:output:0*
T0*0
_output_shapes
:?????????S?*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*0
_output_shapes
:?????????S?2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2a_layer_call_and_return_conditional_losses_583186

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????-::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????-
 
_user_specified_nameinputs
?	
?
B__inference_dense2_layer_call_and_return_conditional_losses_584152

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
C
'__inference_pool1c_layer_call_fn_583059

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_pool1c_layer_call_and_return_conditional_losses_5830532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????:
dense20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAFAAAAQwAAAHNIAAAAfABkAWQCZABkA4UDZgIZAH0BfABkAWQEZABk\nA4UDZgIZAH0CdABqAXwBfAJnAmQFZAaNAn0AdABqAqADfABkB6ECfQB8AFMAKQhOLukBAAAA6QMA\nAADpAgAAAOn/////KQHaBGF4aXMpAulTAAAA6bMAAAApBNoCdGbaBmNvbmNhdNoFaW1hZ2XaBnJl\nc2l6ZSkD2gNpbWfaCnNhdHVyYXRpb27aBXZhbHVlqQByDwAAAPpGQzovVXNlcnMvODA1MS9EZXNr\ndG9wL2Nsb25ldGVzdC9TaGluTmlzaGltdXJhX0ZZUF9KZXRzb25fTmFuby9tb2RlbC5wedoLcHJv\nY2Vzc19pbWcLAAAAcwoAAAAAARIBEgESAQ4B\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "process", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1a", "inbound_nodes": [[["process", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 42, 90, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1b", "inbound_nodes": [[["conv1a", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 45, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1c", "inbound_nodes": [[["conv1b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2a", "inbound_nodes": [[["pool1c", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv2a", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2b", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2c", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["conv2c", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 12, 64]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2d", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 6, 64]}, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pool2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1152]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAFAAAAQwAAAHNIAAAAfABkAWQCZABkA4UDZgIZAH0BfABkAWQEZABk\nA4UDZgIZAH0CdABqAXwBfAJnAmQFZAaNAn0AdABqAqADfABkB6ECfQB8AFMAKQhOLukBAAAA6QMA\nAADpAgAAAOn/////KQHaBGF4aXMpAulTAAAA6bMAAAApBNoCdGbaBmNvbmNhdNoFaW1hZ2XaBnJl\nc2l6ZSkD2gNpbWfaCnNhdHVyYXRpb27aBXZhbHVlqQByDwAAAPpGQzovVXNlcnMvODA1MS9EZXNr\ndG9wL2Nsb25ldGVzdC9TaGluTmlzaGltdXJhX0ZZUF9KZXRzb25fTmFuby9tb2RlbC5wedoLcHJv\nY2Vzc19pbWcLAAAAcwoAAAAAARIBEgESAQ4B\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "process", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1a", "inbound_nodes": [[["process", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 42, 90, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1b", "inbound_nodes": [[["conv1a", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 45, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1c", "inbound_nodes": [[["conv1b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2a", "inbound_nodes": [[["pool1c", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv2a", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2b", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2c", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["conv2c", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 12, 64]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2d", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 6, 64]}, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pool2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1152]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense2", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "process", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAMAAAAFAAAAQwAAAHNIAAAAfABkAWQCZABkA4UDZgIZAH0BfABkAWQEZABk\nA4UDZgIZAH0CdABqAXwBfAJnAmQFZAaNAn0AdABqAqADfABkB6ECfQB8AFMAKQhOLukBAAAA6QMA\nAADpAgAAAOn/////KQHaBGF4aXMpAulTAAAA6bMAAAApBNoCdGbaBmNvbmNhdNoFaW1hZ2XaBnJl\nc2l6ZSkD2gNpbWfaCnNhdHVyYXRpb27aBXZhbHVlqQByDwAAAPpGQzovVXNlcnMvODA1MS9EZXNr\ndG9wL2Nsb25ldGVzdC9TaGluTmlzaGltdXJhX0ZZUF9KZXRzb25fTmFuby9tb2RlbC5wedoLcHJv\nY2Vzc19pbWcLAAAAcwoAAAAAARIBEgESAQ4B\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv1a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 83, 179, 2]}}
?


kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv1b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 42, 90, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 42, 90, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42, 90, 24]}}
?
%trainable_variables
&regularization_losses
'	variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "pool1c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 45, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 21, 45, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 45, 24]}}
?
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?


3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 23, 32]}}
?
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?


=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11, 23, 32]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 23, 32]}}
?
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling2D", "name": "pool2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 12, 64]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "pool2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 12, 64]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 6, 64]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 6, 64]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1152]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1152]}, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
0
1
2
 3
)4
*5
36
47
=8
>9
O10
P11
U12
V13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
 3
)4
*5
36
47
=8
>9
O10
P11
U12
V13"
trackable_list_wrapper
?
[layer_regularization_losses
\metrics
trainable_variables
regularization_losses
]non_trainable_variables

^layers
	variables
_layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
`layer_regularization_losses
ametrics
trainable_variables
regularization_losses
bnon_trainable_variables

clayers
	variables
dlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv1a/kernel
:2conv1a/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
elayer_regularization_losses
fmetrics
trainable_variables
regularization_losses
gnon_trainable_variables

hlayers
	variables
ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv1b/kernel
:2conv1b/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
jlayer_regularization_losses
kmetrics
!trainable_variables
"regularization_losses
lnon_trainable_variables

mlayers
#	variables
nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
olayer_regularization_losses
pmetrics
%trainable_variables
&regularization_losses
qnon_trainable_variables

rlayers
'	variables
slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2a/kernel
: 2conv2a/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
tlayer_regularization_losses
umetrics
+trainable_variables
,regularization_losses
vnon_trainable_variables

wlayers
-	variables
xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ylayer_regularization_losses
zmetrics
/trainable_variables
0regularization_losses
{non_trainable_variables

|layers
1	variables
}layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%  2conv2b/kernel
: 2conv2b/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
~layer_regularization_losses
metrics
5trainable_variables
6regularization_losses
?non_trainable_variables
?layers
7	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
9trainable_variables
:regularization_losses
?non_trainable_variables
?layers
;	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% @2conv2c/kernel
:@2conv2c/bias
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
?trainable_variables
@regularization_losses
?non_trainable_variables
?layers
A	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
Ctrainable_variables
Dregularization_losses
?non_trainable_variables
?layers
E	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
Gtrainable_variables
Hregularization_losses
?non_trainable_variables
?layers
I	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
Ktrainable_variables
Lregularization_losses
?non_trainable_variables
?layers
M	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?	 2dense1/kernel
: 2dense1/bias
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
Qtrainable_variables
Rregularization_losses
?non_trainable_variables
?layers
S	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2dense2/kernel
:2dense2/bias
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?metrics
Wtrainable_variables
Xregularization_losses
?non_trainable_variables
?layers
Y	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
&__inference_model_layer_call_fn_583854
&__inference_model_layer_call_fn_583542
&__inference_model_layer_call_fn_583887
&__inference_model_layer_call_fn_583621?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_583047?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
A__inference_model_layer_call_and_return_conditional_losses_583416
A__inference_model_layer_call_and_return_conditional_losses_583749
A__inference_model_layer_call_and_return_conditional_losses_583462
A__inference_model_layer_call_and_return_conditional_losses_583821?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_process_layer_call_fn_583924
(__inference_process_layer_call_fn_583929?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_process_layer_call_and_return_conditional_losses_583919
C__inference_process_layer_call_and_return_conditional_losses_583903?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv1a_layer_call_fn_583949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1a_layer_call_and_return_conditional_losses_583940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv1b_layer_call_fn_583969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1b_layer_call_and_return_conditional_losses_583960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_pool1c_layer_call_fn_583059?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_pool1c_layer_call_and_return_conditional_losses_583053?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_conv2a_layer_call_fn_583989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2a_layer_call_and_return_conditional_losses_583980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_584011
(__inference_dropout_layer_call_fn_584016?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_584006
C__inference_dropout_layer_call_and_return_conditional_losses_584001?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv2b_layer_call_fn_584036?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2b_layer_call_and_return_conditional_losses_584027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_584058
*__inference_dropout_1_layer_call_fn_584063?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_584053
E__inference_dropout_1_layer_call_and_return_conditional_losses_584048?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv2c_layer_call_fn_584083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2c_layer_call_and_return_conditional_losses_584074?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_584110
*__inference_dropout_2_layer_call_fn_584105?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_584100
E__inference_dropout_2_layer_call_and_return_conditional_losses_584095?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_pool2d_layer_call_fn_583071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_pool2d_layer_call_and_return_conditional_losses_583065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_flatten_layer_call_fn_584121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_584116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense1_layer_call_fn_584141?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense1_layer_call_and_return_conditional_losses_584132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense2_layer_call_fn_584161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense2_layer_call_and_return_conditional_losses_584152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_583656input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_583047} )*34=>OPUV:?7
0?-
+?(
input_1???????????
? "/?,
*
dense2 ?
dense2??????????
B__inference_conv1a_layer_call_and_return_conditional_losses_583940m8?5
.?+
)?&
inputs?????????S?
? "-?*
#? 
0?????????*Z
? ?
'__inference_conv1a_layer_call_fn_583949`8?5
.?+
)?&
inputs?????????S?
? " ??????????*Z?
B__inference_conv1b_layer_call_and_return_conditional_losses_583960l 7?4
-?*
(?%
inputs?????????*Z
? "-?*
#? 
0?????????*Z
? ?
'__inference_conv1b_layer_call_fn_583969_ 7?4
-?*
(?%
inputs?????????*Z
? " ??????????*Z?
B__inference_conv2a_layer_call_and_return_conditional_losses_583980l)*7?4
-?*
(?%
inputs?????????-
? "-?*
#? 
0????????? 
? ?
'__inference_conv2a_layer_call_fn_583989_)*7?4
-?*
(?%
inputs?????????-
? " ?????????? ?
B__inference_conv2b_layer_call_and_return_conditional_losses_584027l347?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
'__inference_conv2b_layer_call_fn_584036_347?4
-?*
(?%
inputs????????? 
? " ?????????? ?
B__inference_conv2c_layer_call_and_return_conditional_losses_584074l=>7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
'__inference_conv2c_layer_call_fn_584083_=>7?4
-?*
(?%
inputs????????? 
? " ??????????@?
B__inference_dense1_layer_call_and_return_conditional_losses_584132]OP0?-
&?#
!?
inputs??????????	
? "%?"
?
0????????? 
? {
'__inference_dense1_layer_call_fn_584141POP0?-
&?#
!?
inputs??????????	
? "?????????? ?
B__inference_dense2_layer_call_and_return_conditional_losses_584152\UV/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense2_layer_call_fn_584161OUV/?,
%?"
 ?
inputs????????? 
? "???????????
E__inference_dropout_1_layer_call_and_return_conditional_losses_584048l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_584053l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
*__inference_dropout_1_layer_call_fn_584058_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
*__inference_dropout_1_layer_call_fn_584063_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_584095l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_584100l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
*__inference_dropout_2_layer_call_fn_584105_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
*__inference_dropout_2_layer_call_fn_584110_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
C__inference_dropout_layer_call_and_return_conditional_losses_584001l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_584006l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
(__inference_dropout_layer_call_fn_584011_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
(__inference_dropout_layer_call_fn_584016_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
C__inference_flatten_layer_call_and_return_conditional_losses_584116a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????	
? ?
(__inference_flatten_layer_call_fn_584121T7?4
-?*
(?%
inputs?????????@
? "???????????	?
A__inference_model_layer_call_and_return_conditional_losses_583416{ )*34=>OPUVB??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_583462{ )*34=>OPUVB??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_583749z )*34=>OPUVA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_583821z )*34=>OPUVA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_583542n )*34=>OPUVB??
8?5
+?(
input_1???????????
p

 
? "???????????
&__inference_model_layer_call_fn_583621n )*34=>OPUVB??
8?5
+?(
input_1???????????
p 

 
? "???????????
&__inference_model_layer_call_fn_583854m )*34=>OPUVA?>
7?4
*?'
inputs???????????
p

 
? "???????????
&__inference_model_layer_call_fn_583887m )*34=>OPUVA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
B__inference_pool1c_layer_call_and_return_conditional_losses_583053?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_pool1c_layer_call_fn_583059?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_pool2d_layer_call_and_return_conditional_losses_583065?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_pool2d_layer_call_fn_583071?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_process_layer_call_and_return_conditional_losses_583903sA?>
7?4
*?'
inputs???????????

 
p
? ".?+
$?!
0?????????S?
? ?
C__inference_process_layer_call_and_return_conditional_losses_583919sA?>
7?4
*?'
inputs???????????

 
p 
? ".?+
$?!
0?????????S?
? ?
(__inference_process_layer_call_fn_583924fA?>
7?4
*?'
inputs???????????

 
p
? "!??????????S??
(__inference_process_layer_call_fn_583929fA?>
7?4
*?'
inputs???????????

 
p 
? "!??????????S??
$__inference_signature_wrapper_583656? )*34=>OPUVE?B
? 
;?8
6
input_1+?(
input_1???????????"/?,
*
dense2 ?
dense2?????????