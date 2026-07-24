import os 
from mmseg.datasets import PrDataset
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.visualization import SegLocalVisualizer
from mmengine.registry import init_default_scope

# Устанавливаем область поиска "mmseg"
# Благодаря этому при инициализации по конфигу
# объекты (модели, датасеты, пайплайны) ищутся именно в mmsegmentation
init_default_scope('mmseg')


def load_ds() -> PrDataset:
    # Инициализируем путь до корневого каталога нашего проекта 
    mmseg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


    # Эти пути понадобятся для иницализации датасета, 
    # Папка с картинками получится объединением (data_root, data_prefix[img_path])
    # А папка с разметкой — объединением (data_root, data_prefix[seg_map_path])
    data_root = os.path.join(mmseg_root, "dataset_pt")
    data_prefix=dict(img_path=os.path.join("img", "val"), seg_map_path=os.path.join("labels", "val"))


    # Сам датасет отвечает исклчительно за то, чтобы распознать структуру данных и корректно
    # считать метаинформацию 
    # Все остальное передаётся в качестве аргумента pipeline
    # Каждый элемент пайплайна — это конфиг модификатора, реализующего некую операцию
    # Например, LoadImageFromFile обогащает семпл картинкой 
    # А LoadAnnotations обогащает семпл картой сегментации     
    # Каждый модификатор описывается конфигом 
    reading_pipeline = [
        dict(type='LoadImageFromFile'), # Реализован в файле mmsegmentation/mmseg/datasets/transforms/loading.py
        dict(type='LoadAnnotations'), # Реализован в файле mmsegmentation/mmseg/datasets/transforms/loading.py
    ]

    # Создаём наш датасет 
    dataset = PrDataset(
        data_root=data_root, 
        data_prefix=data_prefix, 
        pipeline=reading_pipeline,
        img_suffix=".jpg", # Расширение файлов с картинками
        seg_map_suffix=".png" # Расширение файлов с маской сегментации  
    )
    return dataset 


def plot_sample_demo(ds):    
    print(f"Загружен датасет длиной {len(ds)} элементов")
    save_dir = "practicum_work/supplementary/viz"
    # считываем метаинформацию  
    ds_meta = ds.metainfo

    # Подготовим визуализатор, результат будет в папке viz_outputs
    seg_local_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=0.5,
    )
    # Передаём в визуализатор метаинформацию нашего датасета 
    seg_local_visualizer.dataset_meta = dict(
        classes=ds_meta["classes"],
        palette=ds_meta["palette"]
    )

    # Оборачиваем семпл в структуру SegDataSample, совместимую с визуализатором 
    for i in range(len(ds)):
        orig_sample = ds[i]
        plot_sample = SegDataSample()
        plot_sample.gt_sem_seg = PixelData(data=orig_sample["gt_seg_map"])

        img = orig_sample["img"]
        
        # Получаем оригинальное имя файла (из пути 'img_path' или аналогичного ключа)
        # Если ключа нет, сработает дефолтный вариант с индексом
        img_path = orig_sample.get("img_path", f"sample_{i}.png")
        orig_filename = os.path.basename(img_path)
        
        # Сохраняем в целевую директорию с оригинальным именем
        out_file_path = os.path.join(save_dir, orig_filename)
        
        seg_local_visualizer.add_datasample(
            name="example", 
            image=img,
            data_sample=plot_sample,
            show=False,
            draw_pred=False,
            step=i,         
            out_file=out_file_path
        )



ds = load_ds()
plot_sample_demo(ds)