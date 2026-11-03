# 来源与许可

本子工程使用 AGPL-3.0-only，完整文本见 [LICENSE](LICENSE)。公开接口风格参考不表示与其他厂商的消息二进制兼容。

| 内容 | 固定来源 | 许可与本地文件 |
|---|---|---|
| OpenSprite v4机器人模型和网格 | [open_sprite](https://github.com/BeijingDynamics/open_sprite/tree/35afc2126fc31fa4cf21a2e6061c8bfa6c69726a) | AGPL-3.0；`src/opensprite_sdk/assets/robot/LICENSE`，逐文件SHA见assets/manifest.json |
| 原生CAN执行器和ICM采集/估计器 | [open_sprite_runtime](https://github.com/BeijingDynamics/open_sprite_runtime/tree/4515206c24404e18a2d5c168eccdb7468aea6ab7) | AGPL-3.0-only；`native/vendor/LICENSE`，本地变更见`native/vendor/UPSTREAM.md` |
| TDK ICM45686底层驱动 | 官方runtime所固定的TDK源码 | 原始许可和声明保留在`native/third_party/tdk_icm45686/` |
| SDK2 Python接口参考 | [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206) | BSD-3-Clause；参考checkout仅位于忽略的`.reference/`，不作为发行依赖 |
| SDK2 C++接口参考 | [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2/tree/8e1b927586712edd6a4c29d50dd52dff1e163439) | BSD-3-Clause；不随子工程分发参考checkout |
| MuJoCo桥接结构参考 | [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d) | BSD-3-Clause；不随子工程分发参考checkout |

策略导出器按公开SuperMimic方法的现有训练契约实现ELU编码器/解码器和冻结归一化。相关MIT算法声明保留在 `licenses/algorithm-MIT.txt`。导出checkpoint和动作由使用者显式提供，生成bundle继承其权重和数据许可；本SDK发行目录不包含LAFAN原始数据或舞蹈模型权重。

硬件连接和型号参考 [OpenSprite BOM](https://beijingdynamics.github.io/open_sprite_BOM/)。BOM不替代逐机零位、方向、范围和传感器协议的实际标定。
