"""
   File Name   :   sdv_synthesizer.py
   Author      :   zhuxiaofei22@mails.ucas.ac.cn
   Date：      :   2023/7/18
   Description :
"""
import numpy as np
import pandas as pd
from ctgan import CTGAN, TVAE
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer, CTGANSynthesizer

from cvae import ConditionalTVAE, FixedTVAE, DeltaTVAE, DiffTVAE
from utils import train_model_test

global_positive_gan, global_negative_gan = None, None

ctgan_params = {
    'generator_dim': (128, 128),
    'discriminator_dim': (128, 128),
    'discriminator_steps': 1
}


def synthesizer_init(generator_type, generator_params=None, metadata=None, seed=42):
    generator_params = generator_params if generator_params is not None else {}
    if generator_type == 'TVAE':
        synthesizer = TVAE(**generator_params)
    elif generator_type == 'CTGAN':
        # generator_params['epochs'] = 300
        # generator_params['discriminator_steps'] = 3
        # generator_params['pac'] = 50
        ctgan_params.update(generator_params)
        synthesizer = CTGAN(**ctgan_params)
    elif generator_type == 'GaussianCopula':
        synthesizer = GaussianCopulaSynthesizer(metadata)
    elif generator_type == "CopulaGAN":
        synthesizer = CopulaGANSynthesizer(metadata)
    elif generator_type == 'ConditionalTVAE':
        synthesizer = ConditionalTVAE(**generator_params)
    elif generator_type == 'FixedTVAE':
        synthesizer = FixedTVAE(**generator_params)
    elif generator_type == 'DeltaTVAE':
        synthesizer = DeltaTVAE(**generator_params)
    elif generator_type == 'DiffTVAE':
        synthesizer = DiffTVAE(**generator_params)
    else:
        raise ValueError("unknown generator type")
    synthesizer.set_random_state(seed)
    return synthesizer


def _sdv_synthesis(df, generator_type, synthesis_type, oversample_num=0, seed=42, generator_params=None,
                   positive_ratio=None, data_filter_func=None,
                   init_synthesizer=False, discrete_columns=()):
    assert synthesis_type in ['positive', 'negative']

    if init_synthesizer is True:
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
    else:
        metadata = None

    global global_positive_gan, global_negative_gan
    if synthesis_type == 'positive':
        if global_positive_gan is None or init_synthesizer is True:
            print('positive generator init, {}'.format(generator_type))
            global_positive_gan = synthesizer_init(
                generator_type, metadata=metadata, generator_params=generator_params, seed=seed)
            global_positive_gan.fit(df, discrete_columns=discrete_columns)
            # global_positive_gan.fit(df)
        sample_gan = global_positive_gan
    else:
        if global_negative_gan is None or init_synthesizer is True:
            print('negative generator init, {}'.format(generator_type))
            global_negative_gan = synthesizer_init(
                generator_type, metadata=metadata, generator_params=generator_params, seed=seed)
            global_negative_gan.fit(df, discrete_columns=discrete_columns)
            # global_negative_gan.fit(df)
        sample_gan = global_negative_gan

    sample_gan.set_random_state(seed)

    synthetic_data = generate_data(sample_gan, generator_type, synthesis_type, oversample_num, positive_ratio)
    # 过滤不合适的数据
    if data_filter_func is not None:
        synthetic_data = data_filter_func(synthetic_data)

    return synthetic_data

# 全局字典，用于存储每个类别的合成器
global_class_generators = {}

def _sdv_synthesiss(df, generator_type, class_label, oversample_num=0, seed=42, generator_params=None,
                   positive_ratio=None, data_filter_func=None,
                   init_synthesizer=False, discrete_columns=()):

    if init_synthesizer is True:
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
    else:
        metadata = None

    # 检查该类别的合成器是否存在，如果不存在或需要重新初始化，则进行初始化
    if class_label not in global_class_generators or init_synthesizer is True:
        print(f'{class_label} generator init, {generator_type}')
        generator = synthesizer_init(
            generator_type, metadata=metadata, generator_params=generator_params, seed=seed)
        generator.fit(df, discrete_columns=discrete_columns)
        global_class_generators[class_label] = generator

    sample_gan = global_class_generators[class_label]

    sample_gan.set_random_state(seed)

    synthetic_data = generate_data(sample_gan, generator_type, class_label, oversample_num, positive_ratio)
    # 过滤不合适的数据
    if data_filter_func is not None:
        synthetic_data = data_filter_func(synthetic_data)

    return synthetic_data


def generate_data(sample_gan, generator_type, synthesis_type, oversample_num, positive_ratio):
    if generator_type in ['CTGAN', 'TVAE', 'CopulaGAN', 'FixedTVAE', 'DeltaTVAE', 'DiffTVAE']:
        if positive_ratio is None:
            return sample_gan.sample(oversample_num)
        else:
            pn = int(oversample_num * positive_ratio)
            nn = oversample_num - pn
            synthesis_num = pn if synthesis_type == 'positive' else nn
            return sample_gan.sample(synthesis_num)

    if generator_type == 'ConditionalTVAE':
        pn = int(oversample_num * positive_ratio)
        nn = oversample_num - pn
        syn_p = sample_gan.sample(nn, label=1)
        syn_p['label'] = 1
        syn_n = sample_gan.sample(pn, label=0)
        syn_n['label'] = 0
        synthetic_data = pd.concat([syn_p, syn_n])
        return synthetic_data
    raise ValueError(generator_type, "unknown generator %s" % generator_type)


def sdv_synthesis(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                  init_synthesizer=False, discrete_columns=()):
    ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]

    fraud_df = x_train[np.where(y_train > 0.5, True, False)]
    non_fraud_df = x_train[~np.where(y_train > 0.5, True, False)]

    # 使用ctgan合成，针对每一个类都进行一次训练
    fraud_sample_num = int(oversample_num * ratio)
    no_fraud_sample_num = oversample_num - fraud_sample_num

    fraud_df = _sdv_synthesis(
        fraud_df, generator_type, 'positive', oversample_num=fraud_sample_num, seed=seed,
        init_synthesizer=init_synthesizer, discrete_columns=discrete_columns)
    non_fraud_df = _sdv_synthesis(
        non_fraud_df, generator_type, 'negative', oversample_num=no_fraud_sample_num, seed=seed,
        init_synthesizer=init_synthesizer, discrete_columns=discrete_columns)

    fraud_df['label'] = 1
    non_fraud_df['label'] = 0

    combined_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(keep='first')
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    features = [f for f in shuffled_df.columns if f not in ['label']]
    x_train, y_train = shuffled_df[features], shuffled_df['label']
    return x_train, y_train

def sdv_synthesiss(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                  init_synthesizer=False, discrete_columns=()):
    # 统计每个类别的样本数量
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    if len(unique_classes) != 2:
        raise ValueError("输入数据必须是二分类数据，即包含二个不同的类别标签。")
    
    # 计算每个类别的比例
    total_samples = len(y_train)
    class_ratios = {cls: count / total_samples for cls, count in zip(unique_classes, class_counts)}
    
    # 计算每个类别需要合成的样本数量
    class_sample_nums = {cls: int(oversample_num * ratio) for cls, ratio in class_ratios.items()}
    
    # 对每个类别进行数据合成
    synthesized_dfs = []
    for cls in unique_classes:
        class_df = x_train[y_train == cls]
        synthesized_class_df = _sdv_synthesiss(
            class_df, generator_type, f'class_{cls}', oversample_num=class_sample_nums[cls], 
            seed=seed, init_synthesizer=init_synthesizer, discrete_columns=discrete_columns
        )
        synthesized_class_df['label'] = cls
        synthesized_dfs.append(synthesized_class_df)
    
    # 合并所有合成的数据
    combined_df = pd.concat(synthesized_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(keep='first')
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    features = [f for f in shuffled_df.columns if f not in ['label']]
    x_train, y_train = shuffled_df[features], shuffled_df['label']
    return x_train, y_train

def sdv_synthesis_one_gan(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                          init_synthesizer=False):
    # ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]
    x_train['label'] = y_train
    features = [f for f in x_train.columns if f not in ['label']]

    df = _sdv_synthesis(x_train, generator_type, 'positive', oversample_num=oversample_num, seed=seed,
                        init_synthesizer=init_synthesizer, discrete_columns=('label',))
    _x_train, _y_train = df[features], df['label']

    return _x_train, _y_train


def sdv_synthesis_cvae(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                       discrete_columns=(), init_synthesizer=False):
    ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]
    x_train['label'] = y_train
    features = [f for f in x_train.columns if f not in ['label']]

    df = _sdv_synthesis(x_train, generator_type, 'positive', oversample_num=oversample_num, seed=seed,
                        positive_ratio=ratio, discrete_columns=discrete_columns,
                        init_synthesizer=init_synthesizer)
    _x_train, _y_train = df[features], df['label']

    return _x_train, _y_train


def sdv_synthesis_fixed_tvae(x_train, y_train, generator_type, oversample_num=1000, seed=42, positive_ratio=None,
                             init_synthesizer=False):
    ratio = positive_ratio if positive_ratio is not None else np.sum(y_train) / y_train.shape[0]
    x_train['label'] = y_train
    features = [f for f in x_train.columns if f not in ['label']]

    df = _sdv_synthesis(x_train, generator_type, 'positive', oversample_num=oversample_num, seed=seed,
                        positive_ratio=ratio,
                        init_synthesizer=init_synthesizer)
    _x_train, _y_train = df[features], df['label']

    return _x_train, _y_train


def train_with_sdv(x_train, y_train, x_test, y_test,
                   oversample_num=1000, model_name='xgb', data_type='tabular', generator_type='TVAE', seed=42):
    if oversample_num > y_train.shape[0]:
        x_train, y_train = sdv_synthesis(x_train, y_train, generator_type, seed=seed)

    return train_model_test(x_train, y_train, x_test, y_test,
                            model_name=model_name, data_type=data_type)


def sdv_synthesiss3(x_train, y_train, generator_type, oversample_num=1000, seed=42, class_ratios=None,
                  init_synthesizer=False, discrete_columns=()):
    """
    支持多分类的SDV数据合成方法
    
    参数:
    x_train -- 特征数据 (DataFrame)
    y_train -- 标签数据 (Series)
    generator_type -- SDV生成器类型
    oversample_num -- 总合成样本数 (默认1000)
    seed -- 随机种子 (默认42)
    class_ratios -- 自定义类别比例字典，格式: {class_label: ratio} (可选)
    init_synthesizer -- 是否初始化新生成器 (默认False)
    discrete_columns -- 离散特征列名列表 (默认空)
    
    返回:
    x_train_resampled, y_train_resampled -- 合成后的平衡数据
    """
    
    # 获取类别信息
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        raise ValueError("输入数据必须包含至少两个不同类别")
    
    # 计算或验证类别比例
    if class_ratios:
        # 验证自定义比例
        if not isinstance(class_ratios, dict):
            raise TypeError("class_ratios 必须是字典类型")
        if set(class_ratios.keys()) != set(unique_classes):
            raise ValueError("class_ratios 必须包含所有类别标签")
        if not np.isclose(sum(class_ratios.values()), 1.0):
            raise ValueError("类别比例之和必须等于1")
    else:
        # 自动计算原始比例
        total_samples = len(y_train)
        class_ratios = {cls: count / total_samples for cls, count in zip(unique_classes, class_counts)}
    
    # 计算每个类别的合成样本数量
    class_sample_nums = {
        cls: int(oversample_num * ratio)
        for cls, ratio in class_ratios.items()
    }
    
    # 合成数据生成
    synthesized_dfs = []
    for cls in unique_classes:
        class_df = x_train[y_train == cls]
        
        # 如果该类别没有样本，跳过合成
        if len(class_df) == 0:
            continue
            
        synthesized_class_df = _sdv_synthesiss(
            class_df, generator_type, 
            f'class_{cls}', 
            oversample_num=class_sample_nums[cls],
            seed=seed,
            init_synthesizer=init_synthesizer,
            discrete_columns=discrete_columns
        )
        synthesized_class_df['label'] = cls
        synthesized_dfs.append(synthesized_class_df)
    
    # 合并与后处理
    if synthesized_dfs:
        combined_df = pd.concat(synthesized_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(keep='first')
        shuffled_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        features = [f for f in shuffled_df.columns if f != 'label']
        x_train_resampled, y_train_resampled = shuffled_df[features], shuffled_df['label']
    else:
        x_train_resampled, y_train_resampled = x_train.copy(), y_train.copy()
    
    return x_train_resampled, y_train_resampled