# 项目简介
* 用于暗水印的深度学习方法
* UDH(Universal deep hide)方法,加入传统方法LSB,修改Loss函数。
* 使用数据集为 __DIV2K_train_HR__
* 训练代码未提供
* 测试代码在test中
* 所需库文件已如requirements.txt所述
* 由于上传限制，不上传水印提取权重模型


## 测试
* 运行test_dataset.py，使用 __DIV2K_valid_HR__ 验证集。查看水印嵌入效果和提取效果。数据集没有放全.\
效果为cover,container,secret,out_secret输出到test/save_result文件夹中
* 运行test_film.py  ,面向输入为录屏软件输出，较慢，图像的结果会保存到test_picture文件夹下11.png.需要录屏软件的录屏质量较高
* 运行test_watermark.py     有Hide_pic_port(src)，Reveal_one_pic(src)两个函数分别为嵌入和提取水印。输入为单张图像。\
若为嵌入水印，要求图像长宽被16整除，以符合Unet的输入格式。\
可以通过is_cuda选项选择是否使用GPU进行计算

## 远程桌面测试
* remote-desktop-master为实现简易远程桌面代码，为开源代码实现，源链接不明。server.py为服务器端，client.py为用户端。在服务器端口显示client的远程桌面。

* 其中，server.py中修改host和port改为本机的ip地址和port，其中port为闲置端口
>host = "10.61.241.207"\
>port = 8001
* 并将config.ini中的ip地址和port改为server.py中的ip地址和端口号。
* 实现效果为将client中的桌面添加水印后，显示在server中的程序窗口中。
* 远程桌面基本流程没有经过优化，但可以进行相当大的优化，使得提前计算水印，并使得嵌入水印时间几乎为0