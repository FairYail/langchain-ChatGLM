import sys

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search

embedder = SentenceModel("shibing624/text2vec-base-chinese")

# Corpus with example sentences
corpus = [
        '每个人都能设置助战弟子吗',
        '为什么七曜锁妖塔中我的弟子没有技能，属性有变化',
        '如何获取专武',
        '妖王召唤进度规则',
        '妖王的邀请链接如何分享',
        '限时转服',
        '限时转服预约时间',
        '宗门复兴',
        '精卫的织素·铁骨怎么提升',
        '精卫的海魂·铁骨在哪里',
        '精卫专武怎么穿戴不了',
        '红色月魂圣匣之三的保底可以累计吗',
        '荒天古镜不见了、在哪查看',
        '如何获得荒天古镜',
        '风起天元',
        '重开、删除账号',
        '为什么购买了68元武祖令/太清令/冰主令没有给神通技能、轮回者',
        '锁妖塔',
        '基础玩法',
        '宗门、宗门大殿',
        '任务（章节、悬赏、每日）',
        '仙盟玩法',
        '仙盟功能',
        '卡顿处理方法',
        '官方道友群',
        '设置',
        '建筑相关问题',
        '常规活动相关',
        '战斗玩法相关',
        '新手活动相关',
        '镇妖涧玩法',
        '观星台玩法',
        '本服聊天功能',
        '角色信息',
        '改名',
        '头像',
        '声音',
        '掌门洞府',
        '铸剑池',
        '炼丹炉',
        '修炼台',
        '天财阁',
        '宗门大殿',
        '长老靖室',
        '练武场',
        '藏经阁',
        '建造、建筑基本问题',
        '天元斗技大会',
        '落云宗试炼',
        '轮回神殿',
        '斗技',
        '妖窟',
        '游历',
        '重开、换区、删除账号',
        '基金',
        '礼包码',
        'VIP',
        '吕洞宾',
        '限时礼包',
        '神通',
        '神魂',
        '专武',
        '化身',
        '伴生物',
        '轮回棋局',
        '聚灵阵',
        '接待访客',
        '弟子',
        '如何弹劾仙主',
        '莲莲有鱼没办法放置',
        '第二阶段战斗规则',
        '章节任务到40章没有任务了',
        '章节任务已建造了建筑为什么还是显示未完成',
        '为什么我有轮回异宝不能使用',
        '为什么我没有春节活动日榜排行榜奖励',
        '如何解锁铸剑池',
        '妖兽内丹是什么',
        '招收弟子按钮不见了',
        '如何开启仙盟功能',
        '镇妖涧玩法基础介绍',
        '如何进入镇妖涧',
        '如何参与妖王挑战',
        '妖王协助有限制吗',
        '妖王开启方式',
        '妖王分身的刷新时间',
        '第一阶段战斗规则',
        '挑战次数怎么获得',
        '当天没使用的次数可以累计吗',
        '第一阶段的参与条件',
        '观星台玩法基础介绍',
        '观星台的四象是什么',
        '退出仙盟后观星台还会生效吗',
        '如何进入观星台',
        '仙盟等级有什么作用',
        '仙盟职位都有什么',
        '仙盟聊天',
        '如何查询注册时间',
        '分享无法完成',
        '退款',
        '未成年人退款',
        '开挂',
        '开发票',
        '什么条件可以参与天元斗技大会',
        '网络卡顿',
        '谢谢',
        '再见',
        '建议',
        '辛苦了',
        '如何关闭游戏内置声音',
        '道点不增加',
        '游戏画面卡顿',
        '为什么买了基金却还是提示未购买',
        '同在一个微信群里的好友，为什么在游戏里无法和他聊天?',
        '本服是什么意思',
        '为什么不能通过服务器列表进入其他服务器',
        '如何开启私聊功能（传音)',
        '如何开启世界聊天功能（仙境)',
        '为什么不能点击斗技榜单中对方的头像进行聊天',
        '如何给掌门进行改名',
        '如何获得掌门密令',
        '如何给宗门进行改名',
        '如何给弟子改名',
        '如何给轮回者改名',
        '如何获得宗门密令',
        '头像改不了',
        '礼包码兑换不了',
        '掌门能够变成轮回者吗',
        '掌门的记忆可以恢复吗',
        '可以改变掌门性别吗',
        '掌门可以更换吗',
        '掌门洞府可以收起来吗',
        '九凤装备图纸怎么获得',
        '千里蟹装备图纸怎么获得',
        '相柳装备图纸怎么获得',
        '装备图纸有什么用',
        '分解错装备可以找回吗',
        '升级过的装备分解是全额返还升级材料吗',
        '装备可以分解吗',
        '装备怎么升级',
        'BOSS材料从哪里获得',
        '有材料不能铸造',
        '怎么铸造更高阶的装备',
        '器师有什么作用',
        '游历中不同阶级铸器材料分别能在哪些关卡中获得',
        '怎么炼制更高阶的丹药',
        '丹理有什么作用',
        '丹药产量不一样',
        '有炼丹材料不能炼丹',
        '丹药有什么种类',
        '如何获取炼丹材料',
        '道祖雕像附近修炼没有经验加成',
        '卡在建立36个练功点的任务上，需要宗门多少级才能解锁更多练功点',
        '练功点不够用',
        '弟子无法进入桃花散露台、广寒玉镜台修炼',
        '天财阁弟子怎么不见了',
        '天财阁弟子头上显示问号',
        '大量制作材料无法售卖',
        '天财阁弟子显示路不通',
        '宗门大殿经验产出原理',
        '宗门大殿有弟子也无法进行经验产出',
        '长老加成不显示',
        '6级宗门为什么建立不了长老靖室',
        '为什么更换不了长老',
        '长老无法指派工作',
        '长老加成',
        '无法指派弟子到练武场',
        '如何建造练武场',
        '练武场的作用',
        '获得的功法点与描述的不符',
        '为什么有弟子不能指派到藏经阁中',
        '为什么我有神通残页不能研习',
        '如何建造藏经阁',
        '研习概率',
        '神通满级但仍能获取',
        '丢了一个藏经阁',
        '木工坊怎么升到5级不能升级了',
        '为什么建筑升级显示“剩余时间0秒”',
        '为什么已经建造了的建筑不算在任务计数内',
        '灵气值有什么用',
        '路、水道',
        '路不通',
        '风水值降了',
        '天财阁弟子显示没有路',
        '有的建筑没有风水',
        '移动、收起建筑',
        '如何提升建筑上限',
        '弟子无法进入特殊修炼台',
        '如何查看角色ID',
        '什么时候开天元斗技大会',
        '怎么没开出8888机缘',
        '落云宗试炼的达标积分能保留到下次活动吗',
        '红色落云宗密匣的保底能累计吗',
        '如何开启落云宗试炼活动',
        '炼丹没有积分',
        '获取不了通玄子',
        '轮回神殿的落子达标次数会保留到下次活动吗',
        '为什么我之前的落子数没有算在达标里',
        '如何开启轮回神殿活动',
        '为什么常规锁妖塔不奖励神通技能了',
        '常规锁妖塔奖励的技能都有什么',
        '七曜锁妖如何跳层',
        '如何解锁七曜锁妖',
        '为什么木灵界塔不能派木灵根弟子（其它同理）',
        '常规锁妖塔到1500层没有后续',
        '挑战令怎么获得',
        '境界压制会影响神通技能的等级吗',
        '斗技都有什么段位',
        '斗技开放时间',
        '分比对方高但是打不过',
        '斗技场如何进阶',
        '为何买了98元残卷没有立升30级且获得轮回者',
        '如何掉段',
        '掉段条件是什么',
        '参战和助战有什么区别',
        '怎么征召队友',
        '如何进入妖窟',
        '每日免费妖窟刷新',
        '妖窟打完boss超时没有领奖会自动抽奖吗',
        '妖窟掉落奖励',
        '助战没有奖励',
        '游历的蓝色旗帜驻派地在哪',
        '为什么有弟子不能编队到游历编队当中',
        '游历的【自动】按钮是干什么的',
        '游历上方显示的【挑战】字样是什么意思',
        '为什么队伍评分比关卡推荐评分高还打不过',
        '一张地图有多少小关卡',
        '为什么我游历一直在这一关无法继续了',
        '游历获得修为吗',
        '离线后可以自动通关吗',
        '游历点不开',
        '道具获取没有达到灵石收益',
        '妖丹有什么用',
        '灵石收益不够',
        '怎么获得韩天南',
        '风起天元落棋次数不累计',
        '如何开启聚灵阵',
        '聚灵阵可以干什么',
        '后天灵物是什么',
        '如何获取先天灵物',
        '如何获取后天灵物',
        '先天灵物是什么',
        '聚灵阵指派弟子有什么用途',
        '为什么有的轮回者抽取不到',
        '怎么购买轮回棋子',
        '执黑棋和白棋有什么区别',
        '怎么领取图鉴奖励',
        '怎么在轮回商店中购买东西',
        '魂晶怎么获得',
        '必出轮回者次数重置规则',
        '获得重复的轮回者会发生什么',
        '如何穿戴化身',
        '如何卸下化身',
        '错过的化身会返场吗',
        '如何获取还尘珠',
        '如何重置专武',
        '心事',
        '神魂重置后能重复领取图鉴进度奖励吗',
        '如何进行神魂升级',
        '神魂重置',
        '如何获取返魂石',
        '神通技能怎么精炼',
        '神通技能怎么升级',
        '如何获取神通',
        '我已经获取了该神通为什么没有找到',
        '获取了重复神通',
        '没有视频广告',
        '吕洞宾特惠30元、128元礼包',
        '吕洞宾化身',
        '吕洞宾首充',
        '获取神通在轮回商店重复兑换',
        '吕洞宾日礼包',
        '购买了伏雷式礼包找不到伏雷式',
        '雷剑术找不到',
        '礼包码怎么不对',
        '如何添加专属VIP客服',
        '玄像阁雕像在哪领',
        '能帮我看看多少能添加VIP客服吗',
        '添加多次VIP无人回应',
        '活跃度奖励中可领取到的技能都有什么',
        '左下角的任务卷轴点不开了',
        '我怎么没领取到活跃度奖励的技能',
        '获得活跃奖励',
        '弟子达到条件但无法参加悬赏任务',
        '如何生产更高阶的伴生物',
        '灵木所产的伴生物可以使用吗',
        '灵脉产的伴生物有什么用',
        '灵田、药园所产伴生物可以干什么',
        '有伴生物无法炼丹',
        '弟子加成影响伴生物产出吗',
        '伴生物不产出',
        '限时建筑礼包的触发条件和存在时间',
        '限时神通礼包的触发条件和存在时间',
        '限时神通礼包都有哪些',
        '限时建筑礼包都有哪些',
        '我不在线的时候如果有限时礼包怎么办',
        '洪荒至宝如何触发',
        '限时礼包什么时候返场',
        '为什么到14天、40天还没有返场',
        '可以为我单独返场吗',
        '访客给予的灵石、机缘奖励没收到',
        '驻派地解锁的新访客在哪',
        '魔界弟子怎么不刷新了',
        '商人给的道具与等级不匹配',
        '没有访客来了',
        '可以拒绝招收不想要的弟子吗',
        '弟子的境界都有哪些',
        '如何给弟子穿上装备',
        '如何提高弟子的评分',
        '为什么我弟子的异宝不能重置',
        '刚招募的弟子没有看到',
        '如何提升弟子的属性',
        '没有十连招募弟子功能',
        '战斗时弟子怎么站着原地不动',
        '为什么有的弟子不能驻派',
        '弟子界面有小红点，但是没有升级突破神魂升级',
        '弟子消失了',
        '弟子没地方住了',
        '怎么驱逐弟子',
        '无法招募弟子',
        '宗门大殿经验不产出',
        '宗门账簿在哪里看',
        '如何加入官方群',
        '宗门等级到40级不升级',
        '宗门大殿占地方太多了',
        '宗门大殿不升级',
        '提升宗门等级',
        '如何建造长老靖室',
        '宗门大殿不见了',
        '吕洞宾归尘·铁骨怎么提升'
]
corpus_embeddings = embedder.encode(corpus)


while True:
    query = input("Q：")
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))