# 数据挖掘课程大作业

## 一、进入github内部创建个人仓库(repositories)

![image-20240423170629595](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/8dd785f2-1f12-4c43-81d1-fbc519e1fc77)



## 二、点击"New Repository"之后，会进入Git库创建页面，进行选择。


![image-20240423171420508](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/20b7b644-8e16-4f2a-b555-e1db4cde73ce)


1.在**"Repository name"**中填写项目名称，

2.在**Description**中填写项目描述，

3.同时在下面的单选框菜单里选择项目**Public**、**Private**

4.继续下拉滚动条，可以看到一个**"Initialize this repository with a README"**的选项，这个选项表示的是在该项目中，创建一个用于进行项目详细描述的README.md文件，其中包含的是项目的详细描述信息。

5.**Add a license**在使用git作版本控制时，git会默认把git控制的文件夹里面的所有文件都加入到版本控制。

6.**Add .gitignore**，在使用git作版本控制时，git会默认把git控制的文件夹里面的所有文件都加入到版本控制。

7.然后点击**"Create repository"**。这样我们的一个Git库就创建好了，链接为[zhangxuan-thecastle/zxx-DataMining: 数据挖掘作业 (github.com)](https://github.com/zhangxuan-thecastle/zxx-DataMining/tree/main)。




## 三、在文件列表上方，选择“Add file”下拉菜单，然后单击“Upload files”。 也可将文件拖放到浏览器中。

![image-20240423200658499](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/d7bae609-1d41-42e1-aaa5-a5a5faa30c8d)



## 四、获得的计算机技能

### Vit


![image-20240423204329484](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/ec9eb0db-881b-4915-9d9a-b72f9ebb46fa)


ViT将输入图片分为多个patch（16x16），再将每个patch投影为固定长度的向量送入Transformer，后续encoder的操作和原始Transformer中完全相同。但是因为对图片分类，因此在输入序列中加入一个特殊的token，该token对应的输出即为最后的类别预测

**(1) patch embedding：**

例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为**196**，每个patch维度16x16x3=**768**，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符cls，因此最终的维度是**197x768**。到目前为止，已经通过patch embedding将一个视觉问题转化为了一个seq2seq问题

**(2) positional encoding（standard learnable 1D position embeddings）：**

ViT同样需要加入位置编码，位置编码可以理解为一张表，表一共有N行，N的大小和输入序列长度相同，每一行代表一个向量，向量的维度和输入序列embedding的维度相同（768）。注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是**197x768**

**(3) LN/multi-head attention/LN：**

LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是**197x768**

**(4) MLP：**

将维度放大再缩小回去，197x768放大为197x3072，再缩小变为**197x768**



### Transunet

![image-20240115153242117](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/28d5bf64-adf0-4d0e-8019-c96186553e6c)


**首先用 CNN 提取 feature map，然后将patch size设置为 1∗1 ，说白了就是把 feature map 给展开成像素级序列，作为 Transformer 的输入。**

**在 Decoder 阶段，为了弥补 Transformer 在局部信息处理上的不足，又在不同阶段（层）将 CNN encoder 阶段提取的 feature map 和 Transformer hidden feature 结合，这样解码时即有全局信息又有局部细节信息。**

结果：
![1cee27aac02c06e859d116ff42cd925](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/c26d368b-b8eb-41b4-83cf-9045a0ab2fac)



### BRAU

![image-20240423203341511](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/9cb7a096-3de3-4c3f-9cb3-1282a440de55)

本文通过双层路由(`bi-level routing`)提出了一种新颖的**动态稀疏注意力**(`dynamic sparse attention `)，以实现更灵活的**计算分配**和**内容感知**，使其具备动态的查询感知稀疏性，如图(f)所示,先看哪些local window之间是有关系的，确定哪些有关系后再进入到local window里面，让里面每一个小的patch相互做attention（即先粗糙的做一遍attention，觉得哪些值得做，再进入里面细致的做attention）。





### Mamba

并行化——选择性扫描算法

它放弃用卷积来描述SSM，而是定义了一种新的加运算，在并行计算中，连加操作是可并行的，定义新的运算过程:![image-20240421165722085](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/8f365b5c-25a2-4a6b-9881-7dd884481e03)

![image-20240421165729603](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/fdcda4a9-8cc1-4c54-85a1-e22cc445d8ae)

![image-20240421165736255](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/f601fb3a-53d5-4e1b-a936-0b14d5dc99e4)







硬件感知算法

HBM:显卡的高带宽内存

SRAM:显卡的高速缓存区

Transformer 需要把模型各个模块分批次从HBM加载到SRAM去计算，如，先算QKV，再算注意力分数，注意力分数再与输入相乘

Mamba的参数(原始的A,B,C会被直接加载到SRAM，一步直接得到输出，从SRAM写回HBM)


![image-20240421165810248](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/1ef98d2e-90dd-42c3-a459-eca44f6ab0ee)

模型图：

![image-20240421165849251](https://github.com/zhangxuan-thecastle/zxx-DataMining/assets/71864541/fe51972f-6a06-4ae3-a2e9-28857949a146)

将SSM架构比如H3的基础块，transformer中普遍存在的门控MLP相结合，与归一化和残差连接结合，便构成了Mamba架构
