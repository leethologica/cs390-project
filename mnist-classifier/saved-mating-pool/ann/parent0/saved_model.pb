��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
�
dense_58305/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*#
shared_namedense_58305/kernel
{
&dense_58305/kernel/Read/ReadVariableOpReadVariableOpdense_58305/kernel* 
_output_shapes
:
��*
dtype0
y
dense_58305/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_namedense_58305/bias
r
$dense_58305/bias/Read/ReadVariableOpReadVariableOpdense_58305/bias*
_output_shapes	
:�*
dtype0
�
dense_58306/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *#
shared_namedense_58306/kernel
z
&dense_58306/kernel/Read/ReadVariableOpReadVariableOpdense_58306/kernel*
_output_shapes
:	� *
dtype0
x
dense_58306/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namedense_58306/bias
q
$dense_58306/bias/Read/ReadVariableOpReadVariableOpdense_58306/bias*
_output_shapes
: *
dtype0
�
dense_58307/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*#
shared_namedense_58307/kernel
y
&dense_58307/kernel/Read/ReadVariableOpReadVariableOpdense_58307/kernel*
_output_shapes

: 
*
dtype0
x
dense_58307/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_58307/bias
q
$dense_58307/bias/Read/ReadVariableOpReadVariableOpdense_58307/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
h


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
 
�

layers
trainable_variables
layer_metrics
non_trainable_variables
	variables
metrics
 layer_regularization_losses
regularization_losses
 
^\
VARIABLE_VALUEdense_58305/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_58305/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
�

!layers
trainable_variables
"layer_metrics
#non_trainable_variables
	variables
$metrics
%layer_regularization_losses
regularization_losses
^\
VARIABLE_VALUEdense_58306/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_58306/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�

&layers
trainable_variables
'layer_metrics
(non_trainable_variables
	variables
)metrics
*layer_regularization_losses
regularization_losses
^\
VARIABLE_VALUEdense_58307/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_58307/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�

+layers
trainable_variables
,layer_metrics
-non_trainable_variables
	variables
.metrics
/layer_regularization_losses
regularization_losses

0
1
2
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
�
!serving_default_dense_58305_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_dense_58305_inputdense_58305/kerneldense_58305/biasdense_58306/kerneldense_58306/biasdense_58307/kerneldense_58307/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_42372154
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&dense_58305/kernel/Read/ReadVariableOp$dense_58305/bias/Read/ReadVariableOp&dense_58306/kernel/Read/ReadVariableOp$dense_58306/bias/Read/ReadVariableOp&dense_58307/kernel/Read/ReadVariableOp$dense_58307/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8� **
f%R#
!__inference__traced_save_42372336
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_58305/kerneldense_58305/biasdense_58306/kerneldense_58306/biasdense_58307/kerneldense_58307/bias*
Tin
	2*
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_42372364��
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372062
dense_58305_input
dense_58305_42372046
dense_58305_42372048
dense_58306_42372051
dense_58306_42372053
dense_58307_42372056
dense_58307_42372058
identity��#dense_58305/StatefulPartitionedCall�#dense_58306/StatefulPartitionedCall�#dense_58307/StatefulPartitionedCall�
#dense_58305/StatefulPartitionedCallStatefulPartitionedCalldense_58305_inputdense_58305_42372046dense_58305_42372048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58305_layer_call_and_return_conditional_losses_423719722%
#dense_58305/StatefulPartitionedCall�
#dense_58306/StatefulPartitionedCallStatefulPartitionedCall,dense_58305/StatefulPartitionedCall:output:0dense_58306_42372051dense_58306_42372053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58306_layer_call_and_return_conditional_losses_423719992%
#dense_58306/StatefulPartitionedCall�
#dense_58307/StatefulPartitionedCallStatefulPartitionedCall,dense_58306/StatefulPartitionedCall:output:0dense_58307_42372056dense_58307_42372058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58307_layer_call_and_return_conditional_losses_423720262%
#dense_58307/StatefulPartitionedCall�
IdentityIdentity,dense_58307/StatefulPartitionedCall:output:0$^dense_58305/StatefulPartitionedCall$^dense_58306/StatefulPartitionedCall$^dense_58307/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2J
#dense_58305/StatefulPartitionedCall#dense_58305/StatefulPartitionedCall2J
#dense_58306/StatefulPartitionedCall#dense_58306/StatefulPartitionedCall2J
#dense_58307/StatefulPartitionedCall#dense_58307/StatefulPartitionedCall:[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372178

inputs.
*dense_58305_matmul_readvariableop_resource/
+dense_58305_biasadd_readvariableop_resource.
*dense_58306_matmul_readvariableop_resource/
+dense_58306_biasadd_readvariableop_resource.
*dense_58307_matmul_readvariableop_resource/
+dense_58307_biasadd_readvariableop_resource
identity��
!dense_58305/MatMul/ReadVariableOpReadVariableOp*dense_58305_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dense_58305/MatMul/ReadVariableOp�
dense_58305/MatMulMatMulinputs)dense_58305/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_58305/MatMul�
"dense_58305/BiasAdd/ReadVariableOpReadVariableOp+dense_58305_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dense_58305/BiasAdd/ReadVariableOp�
dense_58305/BiasAddBiasAdddense_58305/MatMul:product:0*dense_58305/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_58305/BiasAdd�
!dense_58306/MatMul/ReadVariableOpReadVariableOp*dense_58306_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02#
!dense_58306/MatMul/ReadVariableOp�
dense_58306/MatMulMatMuldense_58305/BiasAdd:output:0)dense_58306/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_58306/MatMul�
"dense_58306/BiasAdd/ReadVariableOpReadVariableOp+dense_58306_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"dense_58306/BiasAdd/ReadVariableOp�
dense_58306/BiasAddBiasAdddense_58306/MatMul:product:0*dense_58306/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_58306/BiasAdd|
dense_58306/ReluReludense_58306/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_58306/Relu�
!dense_58307/MatMul/ReadVariableOpReadVariableOp*dense_58307_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype02#
!dense_58307/MatMul/ReadVariableOp�
dense_58307/MatMulMatMuldense_58306/Relu:activations:0)dense_58307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_58307/MatMul�
"dense_58307/BiasAdd/ReadVariableOpReadVariableOp+dense_58307_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"dense_58307/BiasAdd/ReadVariableOp�
dense_58307/BiasAddBiasAdddense_58307/MatMul:product:0*dense_58307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_58307/BiasAdd�
dense_58307/SoftmaxSoftmaxdense_58307/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_58307/Softmaxq
IdentityIdentitydense_58307/Softmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������:::::::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372043
dense_58305_input
dense_58305_42371983
dense_58305_42371985
dense_58306_42372010
dense_58306_42372012
dense_58307_42372037
dense_58307_42372039
identity��#dense_58305/StatefulPartitionedCall�#dense_58306/StatefulPartitionedCall�#dense_58307/StatefulPartitionedCall�
#dense_58305/StatefulPartitionedCallStatefulPartitionedCalldense_58305_inputdense_58305_42371983dense_58305_42371985*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58305_layer_call_and_return_conditional_losses_423719722%
#dense_58305/StatefulPartitionedCall�
#dense_58306/StatefulPartitionedCallStatefulPartitionedCall,dense_58305/StatefulPartitionedCall:output:0dense_58306_42372010dense_58306_42372012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58306_layer_call_and_return_conditional_losses_423719992%
#dense_58306/StatefulPartitionedCall�
#dense_58307/StatefulPartitionedCallStatefulPartitionedCall,dense_58306/StatefulPartitionedCall:output:0dense_58307_42372037dense_58307_42372039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58307_layer_call_and_return_conditional_losses_423720262%
#dense_58307/StatefulPartitionedCall�
IdentityIdentity,dense_58307/StatefulPartitionedCall:output:0$^dense_58305/StatefulPartitionedCall$^dense_58306/StatefulPartitionedCall$^dense_58307/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2J
#dense_58305/StatefulPartitionedCall#dense_58305/StatefulPartitionedCall2J
#dense_58306/StatefulPartitionedCall#dense_58306/StatefulPartitionedCall2J
#dense_58307/StatefulPartitionedCall#dense_58307/StatefulPartitionedCall:[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
I__inference_dense_58307_layer_call_and_return_conditional_losses_42372286

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference__traced_restore_42372364
file_prefix'
#assignvariableop_dense_58305_kernel'
#assignvariableop_1_dense_58305_bias)
%assignvariableop_2_dense_58306_kernel'
#assignvariableop_3_dense_58306_bias)
%assignvariableop_4_dense_58307_kernel'
#assignvariableop_5_dense_58307_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp#assignvariableop_dense_58305_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_58305_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_58306_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_58306_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_58307_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_58307_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_dense_58306_layer_call_and_return_conditional_losses_42371999

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
!__inference__traced_save_42372336
file_prefix1
-savev2_dense_58305_kernel_read_readvariableop/
+savev2_dense_58305_bias_read_readvariableop1
-savev2_dense_58306_kernel_read_readvariableop/
+savev2_dense_58306_bias_read_readvariableop1
-savev2_dense_58307_kernel_read_readvariableop/
+savev2_dense_58307_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1ecadeb4b1664fb6a57c8177a523705e/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dense_58305_kernel_read_readvariableop+savev2_dense_58305_bias_read_readvariableop-savev2_dense_58306_kernel_read_readvariableop+savev2_dense_58306_bias_read_readvariableop-savev2_dense_58307_kernel_read_readvariableop+savev2_dense_58307_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*K
_input_shapes:
8: :
��:�:	� : : 
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: 
: 

_output_shapes
:
:

_output_shapes
: 
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372202

inputs.
*dense_58305_matmul_readvariableop_resource/
+dense_58305_biasadd_readvariableop_resource.
*dense_58306_matmul_readvariableop_resource/
+dense_58306_biasadd_readvariableop_resource.
*dense_58307_matmul_readvariableop_resource/
+dense_58307_biasadd_readvariableop_resource
identity��
!dense_58305/MatMul/ReadVariableOpReadVariableOp*dense_58305_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dense_58305/MatMul/ReadVariableOp�
dense_58305/MatMulMatMulinputs)dense_58305/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_58305/MatMul�
"dense_58305/BiasAdd/ReadVariableOpReadVariableOp+dense_58305_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dense_58305/BiasAdd/ReadVariableOp�
dense_58305/BiasAddBiasAdddense_58305/MatMul:product:0*dense_58305/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_58305/BiasAdd�
!dense_58306/MatMul/ReadVariableOpReadVariableOp*dense_58306_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02#
!dense_58306/MatMul/ReadVariableOp�
dense_58306/MatMulMatMuldense_58305/BiasAdd:output:0)dense_58306/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_58306/MatMul�
"dense_58306/BiasAdd/ReadVariableOpReadVariableOp+dense_58306_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"dense_58306/BiasAdd/ReadVariableOp�
dense_58306/BiasAddBiasAdddense_58306/MatMul:product:0*dense_58306/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_58306/BiasAdd|
dense_58306/ReluReludense_58306/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_58306/Relu�
!dense_58307/MatMul/ReadVariableOpReadVariableOp*dense_58307_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype02#
!dense_58307/MatMul/ReadVariableOp�
dense_58307/MatMulMatMuldense_58306/Relu:activations:0)dense_58307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_58307/MatMul�
"dense_58307/BiasAdd/ReadVariableOpReadVariableOp+dense_58307_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02$
"dense_58307/BiasAdd/ReadVariableOp�
dense_58307/BiasAddBiasAdddense_58307/MatMul:product:0*dense_58307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_58307/BiasAdd�
dense_58307/SoftmaxSoftmaxdense_58307/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_58307/Softmaxq
IdentityIdentitydense_58307/Softmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������:::::::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_sequential_19435_layer_call_fn_42372135
dense_58305_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_58305_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_sequential_19435_layer_call_and_return_conditional_losses_423721202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
3__inference_sequential_19435_layer_call_fn_42372099
dense_58305_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_58305_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_sequential_19435_layer_call_and_return_conditional_losses_423720842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
3__inference_sequential_19435_layer_call_fn_42372219

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_sequential_19435_layer_call_and_return_conditional_losses_423720842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_dense_58305_layer_call_fn_42372255

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58305_layer_call_and_return_conditional_losses_423719722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_42372154
dense_58305_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_58305_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_423719582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372120

inputs
dense_58305_42372104
dense_58305_42372106
dense_58306_42372109
dense_58306_42372111
dense_58307_42372114
dense_58307_42372116
identity��#dense_58305/StatefulPartitionedCall�#dense_58306/StatefulPartitionedCall�#dense_58307/StatefulPartitionedCall�
#dense_58305/StatefulPartitionedCallStatefulPartitionedCallinputsdense_58305_42372104dense_58305_42372106*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58305_layer_call_and_return_conditional_losses_423719722%
#dense_58305/StatefulPartitionedCall�
#dense_58306/StatefulPartitionedCallStatefulPartitionedCall,dense_58305/StatefulPartitionedCall:output:0dense_58306_42372109dense_58306_42372111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58306_layer_call_and_return_conditional_losses_423719992%
#dense_58306/StatefulPartitionedCall�
#dense_58307/StatefulPartitionedCallStatefulPartitionedCall,dense_58306/StatefulPartitionedCall:output:0dense_58307_42372114dense_58307_42372116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58307_layer_call_and_return_conditional_losses_423720262%
#dense_58307/StatefulPartitionedCall�
IdentityIdentity,dense_58307/StatefulPartitionedCall:output:0$^dense_58305/StatefulPartitionedCall$^dense_58306/StatefulPartitionedCall$^dense_58307/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2J
#dense_58305/StatefulPartitionedCall#dense_58305/StatefulPartitionedCall2J
#dense_58306/StatefulPartitionedCall#dense_58306/StatefulPartitionedCall2J
#dense_58307/StatefulPartitionedCall#dense_58307/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_dense_58305_layer_call_and_return_conditional_losses_42372246

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_dense_58306_layer_call_fn_42372275

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58306_layer_call_and_return_conditional_losses_423719992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_dense_58307_layer_call_fn_42372295

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58307_layer_call_and_return_conditional_losses_423720262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
3__inference_sequential_19435_layer_call_fn_42372236

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_sequential_19435_layer_call_and_return_conditional_losses_423721202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_dense_58307_layer_call_and_return_conditional_losses_42372026

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :::O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372084

inputs
dense_58305_42372068
dense_58305_42372070
dense_58306_42372073
dense_58306_42372075
dense_58307_42372078
dense_58307_42372080
identity��#dense_58305/StatefulPartitionedCall�#dense_58306/StatefulPartitionedCall�#dense_58307/StatefulPartitionedCall�
#dense_58305/StatefulPartitionedCallStatefulPartitionedCallinputsdense_58305_42372068dense_58305_42372070*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58305_layer_call_and_return_conditional_losses_423719722%
#dense_58305/StatefulPartitionedCall�
#dense_58306/StatefulPartitionedCallStatefulPartitionedCall,dense_58305/StatefulPartitionedCall:output:0dense_58306_42372073dense_58306_42372075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58306_layer_call_and_return_conditional_losses_423719992%
#dense_58306/StatefulPartitionedCall�
#dense_58307/StatefulPartitionedCallStatefulPartitionedCall,dense_58306/StatefulPartitionedCall:output:0dense_58307_42372078dense_58307_42372080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dense_58307_layer_call_and_return_conditional_losses_423720262%
#dense_58307/StatefulPartitionedCall�
IdentityIdentity,dense_58307/StatefulPartitionedCall:output:0$^dense_58305/StatefulPartitionedCall$^dense_58306/StatefulPartitionedCall$^dense_58307/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2J
#dense_58305/StatefulPartitionedCall#dense_58305/StatefulPartitionedCall2J
#dense_58306/StatefulPartitionedCall#dense_58306/StatefulPartitionedCall2J
#dense_58307/StatefulPartitionedCall#dense_58307/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference__wrapped_model_42371958
dense_58305_input?
;sequential_19435_dense_58305_matmul_readvariableop_resource@
<sequential_19435_dense_58305_biasadd_readvariableop_resource?
;sequential_19435_dense_58306_matmul_readvariableop_resource@
<sequential_19435_dense_58306_biasadd_readvariableop_resource?
;sequential_19435_dense_58307_matmul_readvariableop_resource@
<sequential_19435_dense_58307_biasadd_readvariableop_resource
identity��
2sequential_19435/dense_58305/MatMul/ReadVariableOpReadVariableOp;sequential_19435_dense_58305_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype024
2sequential_19435/dense_58305/MatMul/ReadVariableOp�
#sequential_19435/dense_58305/MatMulMatMuldense_58305_input:sequential_19435/dense_58305/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#sequential_19435/dense_58305/MatMul�
3sequential_19435/dense_58305/BiasAdd/ReadVariableOpReadVariableOp<sequential_19435_dense_58305_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_19435/dense_58305/BiasAdd/ReadVariableOp�
$sequential_19435/dense_58305/BiasAddBiasAdd-sequential_19435/dense_58305/MatMul:product:0;sequential_19435/dense_58305/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$sequential_19435/dense_58305/BiasAdd�
2sequential_19435/dense_58306/MatMul/ReadVariableOpReadVariableOp;sequential_19435_dense_58306_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype024
2sequential_19435/dense_58306/MatMul/ReadVariableOp�
#sequential_19435/dense_58306/MatMulMatMul-sequential_19435/dense_58305/BiasAdd:output:0:sequential_19435/dense_58306/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2%
#sequential_19435/dense_58306/MatMul�
3sequential_19435/dense_58306/BiasAdd/ReadVariableOpReadVariableOp<sequential_19435_dense_58306_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_19435/dense_58306/BiasAdd/ReadVariableOp�
$sequential_19435/dense_58306/BiasAddBiasAdd-sequential_19435/dense_58306/MatMul:product:0;sequential_19435/dense_58306/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2&
$sequential_19435/dense_58306/BiasAdd�
!sequential_19435/dense_58306/ReluRelu-sequential_19435/dense_58306/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2#
!sequential_19435/dense_58306/Relu�
2sequential_19435/dense_58307/MatMul/ReadVariableOpReadVariableOp;sequential_19435_dense_58307_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype024
2sequential_19435/dense_58307/MatMul/ReadVariableOp�
#sequential_19435/dense_58307/MatMulMatMul/sequential_19435/dense_58306/Relu:activations:0:sequential_19435/dense_58307/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2%
#sequential_19435/dense_58307/MatMul�
3sequential_19435/dense_58307/BiasAdd/ReadVariableOpReadVariableOp<sequential_19435_dense_58307_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3sequential_19435/dense_58307/BiasAdd/ReadVariableOp�
$sequential_19435/dense_58307/BiasAddBiasAdd-sequential_19435/dense_58307/MatMul:product:0;sequential_19435/dense_58307/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2&
$sequential_19435/dense_58307/BiasAdd�
$sequential_19435/dense_58307/SoftmaxSoftmax-sequential_19435/dense_58307/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2&
$sequential_19435/dense_58307/Softmax�
IdentityIdentity.sequential_19435/dense_58307/Softmax:softmax:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������:::::::[ W
(
_output_shapes
:����������
+
_user_specified_namedense_58305_input
�
�
I__inference_dense_58305_layer_call_and_return_conditional_losses_42371972

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_dense_58306_layer_call_and_return_conditional_losses_42372266

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
P
dense_58305_input;
#serving_default_dense_58305_input:0����������?
dense_583070
StatefulPartitionedCall:0���������
tensorflow/serving/predict:�z
�"
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
0_default_save_signature
1__call__
*2&call_and_return_all_conditional_losses"� 
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_19435", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_19435", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_58305_input"}}, {"class_name": "Dense", "config": {"name": "dense_58305", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58306", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58307", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_19435", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_58305_input"}}, {"class_name": "Dense", "config": {"name": "dense_58305", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58306", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_58307", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_58305", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58305", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 784, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_58306", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58306", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_58307", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_58307", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�

layers
trainable_variables
layer_metrics
non_trainable_variables
	variables
metrics
 layer_regularization_losses
regularization_losses
1__call__
0_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
&:$
��2dense_58305/kernel
:�2dense_58305/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

!layers
trainable_variables
"layer_metrics
#non_trainable_variables
	variables
$metrics
%layer_regularization_losses
regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
%:#	� 2dense_58306/kernel
: 2dense_58306/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

&layers
trainable_variables
'layer_metrics
(non_trainable_variables
	variables
)metrics
*layer_regularization_losses
regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
$:" 
2dense_58307/kernel
:
2dense_58307/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

+layers
trainable_variables
,layer_metrics
-non_trainable_variables
	variables
.metrics
/layer_regularization_losses
regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
�2�
#__inference__wrapped_model_42371958�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *1�.
,�)
dense_58305_input����������
�2�
3__inference_sequential_19435_layer_call_fn_42372219
3__inference_sequential_19435_layer_call_fn_42372099
3__inference_sequential_19435_layer_call_fn_42372135
3__inference_sequential_19435_layer_call_fn_42372236�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372178
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372043
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372202
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372062�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_dense_58305_layer_call_fn_42372255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_dense_58305_layer_call_and_return_conditional_losses_42372246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_dense_58306_layer_call_fn_42372275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_dense_58306_layer_call_and_return_conditional_losses_42372266�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_dense_58307_layer_call_fn_42372295�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_dense_58307_layer_call_and_return_conditional_losses_42372286�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
?B=
&__inference_signature_wrapper_42372154dense_58305_input�
#__inference__wrapped_model_42371958�
;�8
1�.
,�)
dense_58305_input����������
� "9�6
4
dense_58307%�"
dense_58307���������
�
I__inference_dense_58305_layer_call_and_return_conditional_losses_42372246^
0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
.__inference_dense_58305_layer_call_fn_42372255Q
0�-
&�#
!�
inputs����������
� "������������
I__inference_dense_58306_layer_call_and_return_conditional_losses_42372266]0�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� �
.__inference_dense_58306_layer_call_fn_42372275P0�-
&�#
!�
inputs����������
� "���������� �
I__inference_dense_58307_layer_call_and_return_conditional_losses_42372286\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������

� �
.__inference_dense_58307_layer_call_fn_42372295O/�,
%�"
 �
inputs��������� 
� "����������
�
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372043t
C�@
9�6
,�)
dense_58305_input����������
p

 
� "%�"
�
0���������

� �
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372062t
C�@
9�6
,�)
dense_58305_input����������
p 

 
� "%�"
�
0���������

� �
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372178i
8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������

� �
N__inference_sequential_19435_layer_call_and_return_conditional_losses_42372202i
8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������

� �
3__inference_sequential_19435_layer_call_fn_42372099g
C�@
9�6
,�)
dense_58305_input����������
p

 
� "����������
�
3__inference_sequential_19435_layer_call_fn_42372135g
C�@
9�6
,�)
dense_58305_input����������
p 

 
� "����������
�
3__inference_sequential_19435_layer_call_fn_42372219\
8�5
.�+
!�
inputs����������
p

 
� "����������
�
3__inference_sequential_19435_layer_call_fn_42372236\
8�5
.�+
!�
inputs����������
p 

 
� "����������
�
&__inference_signature_wrapper_42372154�
P�M
� 
F�C
A
dense_58305_input,�)
dense_58305_input����������"9�6
4
dense_58307%�"
dense_58307���������
