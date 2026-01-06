# ML6

Создал 2 модели на распознанование машин и мотоциклов в Aws Sagemaker <br>
<img width="996" height="723" alt="image" src="https://github.com/user-attachments/assets/336784e2-899f-48c8-b91a-93d8eb1448a9" />
<br>

Далее запустил A/B тестирование на 90/10  через Jupiter: <br>

```
# Ячейка 1: Настройка A/B теста для моделей из Canvas
print("=" * 60)
print("🎯 A/B ТЕСТИРОВАНИЕ ДЛЯ МОДЕЛЕЙ ИЗ CANVAS")
print("=" * 60)

# 1. Импорт библиотек
import sagemaker
import boto3
import json
from datetime import datetime
import time

# 2. Автоматическая настройка
session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name

print(f"📍 Регион: {region}")
print(f"👤 Роль: {role.split('/')[-1]}")
print()

# 3. Ваши ARN моделей из Canvas
CHAMPION_ARN = "arn:aws:sagemaker:us-east-1:139872254153:model/canvas-model-2026-01-05-23-31-30-548896"
CHALLENGER_ARN = "arn:aws:sagemaker:us-east-1:139872254153:model/canvas-model-2026-01-06-00-45-46-930566"

# Извлекаем ModelName из ARN
def extract_model_name(arn):
    return arn.split(':')[-1].split('/')[-1]

champion_model_name = extract_model_name(CHAMPION_ARN)
challenger_model_name = extract_model_name(CHALLENGER_ARN)

print(f"📋 Модели:")
print(f"   Champion: {champion_model_name}")
print(f"   Challenger: {challenger_model_name}")
print()

# Создаем клиент SageMaker
sm_client = boto3.client('sagemaker')
runtime = boto3.client('runtime.sagemaker')

# Создаём новый быстрый эндпоинт
print("🚀 СОЗДАЁМ БЫСТРЫЙ A/B ЭНДПОИНТ")
print("=" * 50)

# Имена
endpoint_name = "cat-dog-ab-demo"
config_name = "ab-config-demo"

print(f"Эндпоинт: {endpoint_name}")
print(f"Конфигурация: {config_name}")
print()

# Создаём конфигурацию с быстрым инстансом
print("🔧 Создаём конфигурацию...")
try:
    # Используем ml.m5.large - быстрее запускается
    response = sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'champion-variant',
                'ModelName': champion_model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',  # БЫСТРЕЕ!
                'InitialVariantWeight': 90
            },
            {
                'VariantName': 'challenger-variant',
                'ModelName': challenger_model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 10
            }
        ]
    )
    print("✅ Конфигурация создана")
    
    # Создаём эндпоинт
    print("🚀 Запускаем эндпоинт...")
    response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    
    print("✅ Эндпоинт запускается!")
    print(f"   ARN: {response['EndpointArn']}")
    print()
    print("⏳ Ждём 2 минуты...")
    print("   ml.m5.large запускается за 1-2 минуты")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("\nПроверьте что модели существуют:")
    print(f"  Champion: {champion_model_name}")
    print(f"  Challenger: {challenger_model_name}")

```

Проверка работы: <br>
```
# Тестируем распределение 90/10 с детальным выводом
print("🎯 ТЕСТ РАСПРЕДЕЛЕНИЯ 90/10 (50 запросов)")
print("=" * 65)

ENDPOINT_NAME = "cat-dog-ab-demo"

def test_90_10_detailed(num_requests=50):
    """Тестирует распределение 90/10 с детальным выводом"""
    
    results = {'champion': 0, 'challenger': 0, 'errors': 0}
    
    print(f"📤 Отправляем {num_requests} запросов...")
    print("📊 Ожидаем: ~90% Champion (~45), ~10% Challenger (~5)")
    print("-" * 65)
    
    for i in range(num_requests):
        try:
            with open("Bike (31).jpg", "rb") as f:
                image_data = f.read()
            
            # Отправляем запрос
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/x-image',
                Body=image_data,
                Accept='application/json'
            )
            
            # Получаем ответ
            result = json.loads(response['Body'].read().decode())
            predicted_label = result['predicted_label']
            probability = result['probability']
            
            # Определяем модель
            variant = response['ResponseMetadata']['HTTPHeaders'].get(
                'x-amzn-invoked-production-variant', 
                'unknown'
            )
            
            if 'champion' in variant:
                results['champion'] += 1
                model_name = "Champion"
            elif 'challenger' in variant:
                results['challenger'] += 1
                model_name = "Challenger"
            else:
                model_name = "Unknown"
            
            # Выводим результат КАК БЫЛО С 70/30
            print(f"{i+1:2d}. {model_name:10} - предсказание: '{predicted_label}' ({probability:.1%})")
            
        except Exception as e:
            results['errors'] += 1
            print(f"{i+1:2d}. ERROR - {str(e)[:50]}")
    
    print("=" * 65)
    print("📈 ИТОГИ после {num_requests} запросов:")
    print()
    print(f"   Champion:  {results['champion']:2d} запросов ({results['champion']/num_requests*100:.1f}%)")
    print(f"   Challenger: {results['challenger']:2d} запросов ({results['challenger']/num_requests*100:.1f}%)")
    print(f"   Ошибки:    {results['errors']:2d}")
    print()
    
    # Статистический анализ
    print("📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ 90/10:")
    expected_champion = num_requests * 0.9
    expected_challenger = num_requests * 0.1
    
    champ_diff = results['champion'] - expected_champion
    chall_diff = results['challenger'] - expected_challenger
    
    print(f"   Ожидалось Champion: {expected_champion:.1f}, получили: {results['champion']}")
    print(f"   Ожидалось Challenger: {expected_challenger:.1f}, получили: {results['challenger']}")
    print()
    
    if abs(champ_diff) < 7 and abs(chall_diff) < 7:
        print("✅ РАСПРЕДЕЛЕНИЕ 90/10 РАБОТАЕТ ПРАВИЛЬНО!")
        print("   Отклонение в пределах нормы для 50 запросов")
    else:
        print("⚠️  Отклонение от ожидаемого распределения")
        print(f"   Champion: {champ_diff:+.1f} от ожидания")
        print(f"   Challenger: {chall_diff:+.1f} от ожидания")
    
    # Анализ предсказаний
    print()
    print("🎯 АНАЛИЗ ПРЕДСКАЗАНИЙ:")
    print("   Обрати внимание: модели дают РАЗНЫЕ предсказания!")
    print("   Champion → 'Car' (машина)")
    print("   Challenger → 'Bike' (велосипед)")
    print("   Это нормально - модели обучены на разных данных")
    
    return results

# Запускаем тест
print("Начинаем тест 90/10... (займёт ~1-2 минуты)")
results_90_10 = test_90_10_detailed(50)
```
<br>

```
🎯 ТЕСТ РАСПРЕДЕЛЕНИЯ 90/10 (50 запросов)
=================================================================
Начинаем тест 90/10... (займёт ~1-2 минуты)
📤 Отправляем 50 запросов...
📊 Ожидаем: ~90% Champion (~45), ~10% Challenger (~5)
-----------------------------------------------------------------
 1. Champion   - предсказание: 'Bike' (88.2%)
 2. Champion   - предсказание: 'Bike' (88.2%)
 3. Champion   - предсказание: 'Bike' (88.2%)
 4. Champion   - предсказание: 'Bike' (88.2%)
 5. Champion   - предсказание: 'Bike' (88.2%)
 6. Champion   - предсказание: 'Bike' (88.2%)
 7. Champion   - предсказание: 'Bike' (88.2%)
 8. Champion   - предсказание: 'Bike' (88.2%)
 9. Champion   - предсказание: 'Bike' (88.2%)
10. Champion   - предсказание: 'Bike' (88.2%)
11. Champion   - предсказание: 'Bike' (88.2%)
12. Champion   - предсказание: 'Bike' (88.2%)
13. Champion   - предсказание: 'Bike' (88.2%)
14. Champion   - предсказание: 'Bike' (88.2%)
15. Champion   - предсказание: 'Bike' (88.2%)
16. Champion   - предсказание: 'Bike' (88.2%)
17. Champion   - предсказание: 'Bike' (88.2%)
18. Champion   - предсказание: 'Bike' (88.2%)
19. Champion   - предсказание: 'Bike' (88.2%)
20. Challenger - предсказание: 'Bike' (92.5%)
21. Champion   - предсказание: 'Bike' (88.2%)
22. Champion   - предсказание: 'Bike' (88.2%)
23. Champion   - предсказание: 'Bike' (88.2%)
24. Champion   - предсказание: 'Bike' (88.2%)
25. Champion   - предсказание: 'Bike' (88.2%)
26. Champion   - предсказание: 'Bike' (88.2%)
27. Champion   - предсказание: 'Bike' (88.2%)
28. Champion   - предсказание: 'Bike' (88.2%)
29. Challenger - предсказание: 'Bike' (92.5%)
30. Champion   - предсказание: 'Bike' (88.2%)
31. Champion   - предсказание: 'Bike' (88.2%)
32. Champion   - предсказание: 'Bike' (88.2%)
33. Champion   - предсказание: 'Bike' (88.2%)
34. Champion   - предсказание: 'Bike' (88.2%)
35. Champion   - предсказание: 'Bike' (88.2%)
36. Champion   - предсказание: 'Bike' (88.2%)
37. Champion   - предсказание: 'Bike' (88.2%)
38. Champion   - предсказание: 'Bike' (88.2%)
39. Champion   - предсказание: 'Bike' (88.2%)
40. Champion   - предсказание: 'Bike' (88.2%)
41. Champion   - предсказание: 'Bike' (88.2%)
42. Champion   - предсказание: 'Bike' (88.2%)
43. Champion   - предсказание: 'Bike' (88.2%)
44. Champion   - предсказание: 'Bike' (88.2%)
45. Champion   - предсказание: 'Bike' (88.2%)
46. Challenger - предсказание: 'Bike' (92.5%)
47. Champion   - предсказание: 'Bike' (88.2%)
48. Champion   - предсказание: 'Bike' (88.2%)
49. Champion   - предсказание: 'Bike' (88.2%)
50. Champion   - предсказание: 'Bike' (88.2%)
=================================================================
📈 ИТОГИ после {num_requests} запросов:

   Champion:  47 запросов (94.0%)
   Challenger:  3 запросов (6.0%)
   Ошибки:     0

📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ 90/10:
   Ожидалось Champion: 45.0, получили: 47
   Ожидалось Challenger: 5.0, получили: 3

✅ РАСПРЕДЕЛЕНИЕ 90/10 РАБОТАЕТ ПРАВИЛЬНО!
   Отклонение в пределах нормы для 50 запросов

🎯 АНАЛИЗ ПРЕДСКАЗАНИЙ:
   Обрати внимание: модели дают РАЗНЫЕ предсказания!
   Champion → 'Car' (машина)
   Challenger → 'Bike' (велосипед)
   Это нормально - модели обучены на разных данных

```
<br>

Поменял распределение на 70/30: <br>

```
# ЗАДАНИЕ 3 (исправленное): Меняем распределение на 70/30
print("🎯 ЗАДАНИЕ 3: Traffic splitting 90/10 → 70/30 (исправлено)")
print("=" * 60)

ENDPOINT_NAME = "cat-dog-ab-demo"

try:
    # 1. Меняем веса БЕЗ изменения InstanceCount
    print("🔄 Обновляем только веса (без изменения инстансов)...")
    response = sm_client.update_endpoint_weights_and_capacities(
        EndpointName=ENDPOINT_NAME,
        DesiredWeightsAndCapacities=[
            {
                'VariantName': 'champion-variant',
                'DesiredWeight': 70.0,  # Было 90, стало 70
                # НЕ УКАЗЫВАЕМ DesiredInstanceCount
            },
            {
                'VariantName': 'challenger-variant', 
                'DesiredWeight': 30.0,  # Было 10, стало 30
                # НЕ УКАЗЫВАЕМ DesiredInstanceCount
            }
        ]
    )
    
    print("✅ Распределение обновлено!")
    print("   Champion: 70% (было 90%)")
    print("   Challenger: 30% (было 10%)")
    
    # 2. Ждём применения
    print("\n⏳ Ждём 30 секунд для применения изменений...")
    import time
    time.sleep(30)
    
    # 3. Проверяем
    print("\n🔍 Проверяем новое распределение:")
    response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    
    print("   Текущие веса на эндпоинте:")
    for variant in response['ProductionVariants']:
        print(f"   - {variant['VariantName']}: {variant['CurrentWeight']}%")
    
    print("\n🎯 ГОТОВО! Теперь тестируем распределение 70/30")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("\n💡 Альтернативный вариант:")
    print("   Создать новую конфигурацию с весами 70/30")
```

Так же проверил: <br>

```
🎯 ЗАДАНИЕ 4: Тестируем распределение 70/30 (50 запросов)
=================================================================
Начинаем тест... (займёт ~1-2 минуты)
📤 Отправляем 50 запросов...
📊 Ожидаем: ~70% Champion (~35), ~30% Challenger (~15)
-----------------------------------------------------------------
 1. Challenger - предсказание: 'Bike' (65.3%)
 2. Champion   - предсказание: 'Car' (52.4%)
 3. Champion   - предсказание: 'Car' (52.4%)
 4. Champion   - предсказание: 'Car' (52.4%)
 5. Challenger - предсказание: 'Bike' (65.3%)
 6. Champion   - предсказание: 'Car' (52.4%)
 7. Champion   - предсказание: 'Car' (52.4%)
 8. Challenger - предсказание: 'Bike' (65.3%)
 9. Champion   - предсказание: 'Car' (52.4%)
10. Challenger - предсказание: 'Bike' (65.3%)
11. Champion   - предсказание: 'Car' (52.4%)
12. Champion   - предсказание: 'Car' (52.4%)
13. Champion   - предсказание: 'Car' (52.4%)
14. Champion   - предсказание: 'Car' (52.4%)
15. Challenger - предсказание: 'Bike' (65.3%)
16. Champion   - предсказание: 'Car' (52.4%)
17. Champion   - предсказание: 'Car' (52.4%)
18. Challenger - предсказание: 'Bike' (65.3%)
19. Champion   - предсказание: 'Car' (52.4%)
20. Champion   - предсказание: 'Car' (52.4%)
21. Challenger - предсказание: 'Bike' (65.3%)
22. Challenger - предсказание: 'Bike' (65.3%)
23. Champion   - предсказание: 'Car' (52.4%)
24. Champion   - предсказание: 'Car' (52.4%)
25. Champion   - предсказание: 'Car' (52.4%)
26. Champion   - предсказание: 'Car' (52.4%)
27. Champion   - предсказание: 'Car' (52.4%)
28. Champion   - предсказание: 'Car' (52.4%)
29. Challenger - предсказание: 'Bike' (65.3%)
30. Challenger - предсказание: 'Bike' (65.3%)
31. Challenger - предсказание: 'Bike' (65.3%)
32. Champion   - предсказание: 'Car' (52.4%)
33. Champion   - предсказание: 'Car' (52.4%)
34. Challenger - предсказание: 'Bike' (65.3%)
35. Champion   - предсказание: 'Car' (52.4%)
36. Champion   - предсказание: 'Car' (52.4%)
37. Challenger - предсказание: 'Bike' (65.3%)
38. Champion   - предсказание: 'Car' (52.4%)
39. Champion   - предсказание: 'Car' (52.4%)
40. Champion   - предсказание: 'Car' (52.4%)
41. Challenger - предсказание: 'Bike' (65.3%)
42. Champion   - предсказание: 'Car' (52.4%)
43. Champion   - предсказание: 'Car' (52.4%)
44. Challenger - предсказание: 'Bike' (65.3%)
45. Challenger - предсказание: 'Bike' (65.3%)
46. Champion   - предсказание: 'Car' (52.4%)
47. Champion   - предсказание: 'Car' (52.4%)
48. Champion   - предсказание: 'Car' (52.4%)
49. Champion   - предсказание: 'Car' (52.4%)
50. Challenger - предсказание: 'Bike' (65.3%)
=================================================================
📈 ИТОГИ после {num_requests} запросов:

   Champion:  33 запросов (66.0%)
   Challenger: 17 запросов (34.0%)
   Ошибки:     0

📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ:
   Ожидалось Champion: 35.0, получили: 33
   Ожидалось Challenger: 15.0, получили: 17

✅ РАСПРЕДЕЛЕНИЕ 70/30 РАБОТАЕТ ПРАВИЛЬНО!
   Отклонение в пределах нормы
```
<br>

Далее поменял на 50/50 и так же проверил: <br>

```
🎯 ЗАДАНИЕ 4: Тестируем распределение 50/50 (50 запросов)
=================================================================
Начинаем тест... (займёт ~1-2 минуты)
📤 Отправляем 50 запросов...
📊 Ожидаем: ~50% Champion (~25), ~50% Challenger (~25)
-----------------------------------------------------------------
 1. Challenger - предсказание: 'Bike' (65.3%)
 2. Challenger - предсказание: 'Bike' (65.3%)
 3. Champion   - предсказание: 'Car' (52.4%)
 4. Challenger - предсказание: 'Bike' (65.3%)
 5. Champion   - предсказание: 'Car' (52.4%)
 6. Champion   - предсказание: 'Car' (52.4%)
 7. Champion   - предсказание: 'Car' (52.4%)
 8. Challenger - предсказание: 'Bike' (65.3%)
 9. Champion   - предсказание: 'Car' (52.4%)
10. Challenger - предсказание: 'Bike' (65.3%)
11. Champion   - предсказание: 'Car' (52.4%)
12. Champion   - предсказание: 'Car' (52.4%)
13. Challenger - предсказание: 'Bike' (65.3%)
14. Challenger - предсказание: 'Bike' (65.3%)
15. Champion   - предсказание: 'Car' (52.4%)
16. Champion   - предсказание: 'Car' (52.4%)
17. Champion   - предсказание: 'Car' (52.4%)
18. Champion   - предсказание: 'Car' (52.4%)
19. Challenger - предсказание: 'Bike' (65.3%)
20. Champion   - предсказание: 'Car' (52.4%)
21. Challenger - предсказание: 'Bike' (65.3%)
22. Champion   - предсказание: 'Car' (52.4%)
23. Challenger - предсказание: 'Bike' (65.3%)
24. Champion   - предсказание: 'Car' (52.4%)
25. Champion   - предсказание: 'Car' (52.4%)
26. Challenger - предсказание: 'Bike' (65.3%)
27. Champion   - предсказание: 'Car' (52.4%)
28. Champion   - предсказание: 'Car' (52.4%)
29. Champion   - предсказание: 'Car' (52.4%)
30. Champion   - предсказание: 'Car' (52.4%)
31. Challenger - предсказание: 'Bike' (65.3%)
32. Champion   - предсказание: 'Car' (52.4%)
33. Champion   - предсказание: 'Car' (52.4%)
34. Challenger - предсказание: 'Bike' (65.3%)
35. Challenger - предсказание: 'Bike' (65.3%)
36. Challenger - предсказание: 'Bike' (65.3%)
37. Challenger - предсказание: 'Bike' (65.3%)
38. Challenger - предсказание: 'Bike' (65.3%)
39. Champion   - предсказание: 'Car' (52.4%)
40. Champion   - предсказание: 'Car' (52.4%)
41. Challenger - предсказание: 'Bike' (65.3%)
42. Challenger - предсказание: 'Bike' (65.3%)
43. Champion   - предсказание: 'Car' (52.4%)
44. Challenger - предсказание: 'Bike' (65.3%)
45. Champion   - предсказание: 'Car' (52.4%)
46. Champion   - предсказание: 'Car' (52.4%)
47. Challenger - предсказание: 'Bike' (65.3%)
48. Challenger - предсказание: 'Bike' (65.3%)
49. Champion   - предсказание: 'Car' (52.4%)
50. Champion   - предсказание: 'Car' (52.4%)
=================================================================
📈 ИТОГИ после {num_requests} запросов:

   Champion:  28 запросов (56.0%)
   Challenger: 22 запросов (44.0%)
   Ошибки:     0

📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ:
   Ожидалось Champion: 25.0, получили: 28
   Ожидалось Challenger: 25.0, получили: 22


✅ РАСПРЕДЕЛЕНИЕ 50/50 РАБОТАЕТ ПРАВИЛЬНО!
   Отклонение в пределах нормы
```

Сохраннеи метрик: <br>

```
# СИСТЕМА АВТОМАТИЧЕСКОГО ЛОГИРОВАНИЯ МЕТРИК
print("📊 НАСТРОЙКА АВТОМАТИЧЕСКОГО СОХРАНЕНИЯ МЕТРИК")
print("=" * 60)

import json
import csv
from datetime import datetime
import os

class MetricsLogger:
    """Класс для автоматического сохранения метрик"""
    
    def __init__(self, log_dir="metrics_logs"):
        self.log_dir = log_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Создаёт папки для логирования"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"✅ Создана папка для логов: {self.log_dir}")
        
        # Файлы для логов
        self.request_log_file = os.path.join(self.log_dir, "requests.csv")
        self.test_log_file = os.path.join(self.log_dir, "tests.csv")
        self.summary_file = os.path.join(self.log_dir, "summary.json")
        
        # Инициализируем файлы если их нет
        self.init_csv_files()
    
    def init_csv_files(self):
        """Создаёт CSV файлы с заголовками"""
        
        # Файл для отдельных запросов
        if not os.path.exists(self.request_log_file):
            with open(self.request_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'request_id', 'model', 'prediction',
                    'probability', 'latency_ms', 'endpoint', 'status'
                ])
        
        # Файл для тестов
        if not os.path.exists(self.test_log_file):
            with open(self.test_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'test_name', 'distribution', 
                    'total_requests', 'champion_count', 'challenger_count',
                    'champion_percent', 'challenger_percent', 'errors'
                ])
    
    def log_request(self, request_id, model, prediction, probability, 
                   latency_ms, endpoint="cat-dog-ab-demo", status="success"):
        """Логирует один запрос"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'model': model,
            'prediction': prediction,
            'probability': probability,
            'latency_ms': latency_ms,
            'endpoint': endpoint,
            'status': status
        }
        
        # Записываем в CSV
        with open(self.request_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry['timestamp'],
                log_entry['request_id'],
                log_entry['model'],
                log_entry['prediction'],
                log_entry['probability'],
                log_entry['latency_ms'],
                log_entry['endpoint'],
                log_entry['status']
            ])
        
        # Также добавляем в JSON лог
        self._append_to_json_log("requests", log_entry)
        
        return log_entry
    
    def log_test(self, test_name, distribution, results):
        """Логирует результаты теста"""
        
        total = results.get('champion', 0) + results.get('challenger', 0)
        champion_percent = results.get('champion', 0)/total*100 if total > 0 else 0
        challenger_percent = results.get('challenger', 0)/total*100 if total > 0 else 0
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'distribution': distribution,
            'total_requests': total,
            'champion_count': results.get('champion', 0),
            'challenger_count': results.get('challenger', 0),
            'champion_percent': champion_percent,
            'challenger_percent': challenger_percent,
            'errors': results.get('errors', 0)
        }
        
        # Записываем в CSV
        with open(self.test_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry['timestamp'],
                log_entry['test_name'],
                log_entry['distribution'],
                log_entry['total_requests'],
                log_entry['champion_count'],
                log_entry['challenger_count'],
                log_entry['champion_percent'],
                log_entry['challenger_percent'],
                log_entry['errors']
            ])
        
        # Обновляем сводный файл
        self._update_summary(test_name, log_entry)
        
        return log_entry
    
    def _append_to_json_log(self, log_type, entry):
        """Добавляет запись в JSON файл"""
        json_file = os.path.join(self.log_dir, f"{log_type}_log.json")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        
        data.append(entry)
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_summary(self, test_name, test_data):
        """Обновляет сводный файл"""
        try:
            with open(self.summary_file, 'r') as f:
                summary = json.load(f)
        except:
            summary = {}
        
        summary[test_name] = {
            'last_run': datetime.now().isoformat(),
            'data': test_data
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_stats(self):
        """Возвращает статистику по логам"""
        stats = {
            'total_requests': 0,
            'total_tests': 0,
            'last_test': None
        }
        
        try:
            # Считаем запросы
            with open(self.request_log_file, 'r') as f:
                reader = csv.DictReader(f)
                stats['total_requests'] = sum(1 for _ in reader) - 1  # минус заголовок
            
            # Считаем тесты
            with open(self.test_log_file, 'r') as f:
                reader = csv.DictReader(f)
                tests = list(reader)
                stats['total_tests'] = len(tests)
                if tests:
                    stats['last_test'] = tests[-1].get('test_name', 'N/A')
                    
        except Exception as e:
            pass
        
        return stats

# Создаём логгер
logger = MetricsLogger()

print("✅ Система логирования создана!")
print(f"📁 Папка с логами: {os.path.abspath(logger.log_dir)}")
print()

# Показываем статистику
stats = logger.get_stats()
print("📊 СТАТИСТИКА ЛОГОВ:")
print(f"   Всего запросов: {stats['total_requests']}")
print(f"   Всего тестов: {stats['total_tests']}")
print(f"   Последний тест: {stats['last_test'] or 'Нет'}")
```

Тест с логированием: <br>
```
# ОБНОВЛЁННЫЙ ТЕСТ С АВТОМАТИЧЕСКИМ ЛОГИРОВАНИЕМ
print("🎯 ТЕСТ С АВТОМАТИЧЕСКИМ ЛОГИРОВАНИЕМ")
print("=" * 50)

def test_with_logging(num_requests=10, test_name="quick_test"):
    """Тест с автоматическим логированием каждого запроса"""
    
    print(f"📤 Отправляем {num_requests} запросов с логированием...")
    
    results = {'champion': 0, 'challenger': 0, 'errors': 0}
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            
            with open("Bike (1111).jpg", "rb") as f:
                image_data = f.read()
            
            # Отправляем запрос
            response = runtime.invoke_endpoint(
                EndpointName="cat-dog-ab-demo",
                ContentType='application/x-image',
                Body=image_data,
                Accept='application/json'
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Получаем результат
            result = json.loads(response['Body'].read().decode())
            prediction = result['predicted_label']
            probability = result['probability']
            
            # Определяем модель
            variant = response['ResponseMetadata']['HTTPHeaders'].get(
                'x-amzn-invoked-production-variant', 
                'unknown'
            )
            
            if 'champion' in variant:
                model = "champion"
                results['champion'] += 1
            elif 'challenger' in variant:
                model = "challenger"
                results['challenger'] += 1
            else:
                model = "unknown"
            
            # ЛОГИРУЕМ КАЖДЫЙ ЗАПРОС
            logger.log_request(
                request_id=f"{test_name}_{i+1}",
                model=model,
                prediction=prediction,
                probability=probability,
                latency_ms=latency
            )
            
            print(f"  {i+1:2d}: {model:10} → {prediction} ({probability:.1%})")
            
        except Exception as e:
            results['errors'] += 1
            # Логируем ошибку
            logger.log_request(
                request_id=f"{test_name}_error_{i+1}",
                model="error",
                prediction="ERROR",
                probability=0.0,
                latency_ms=0,
                status=f"error: {str(e)[:50]}"
            )
            print(f"  {i+1:2d}: Ошибка ❌")
    
    # Логируем весь тест
    logger.log_test(
        test_name=test_name,
        distribution="current",
        results=results
    )
    
    print()
    print(f"📊 Результаты теста '{test_name}':")
    print(f"   Champion:  {results['champion']}")
    print(f"   Challenger: {results['challenger']}")
    print(f"   Ошибки:    {results['errors']}")
    
    return results

# Запускаем тест с логированием
print("Начинаем тест с автоматическим логированием...")
test_results = test_with_logging(5, "auto_log_test")

print()
print("✅ ВСЁ СОХРАНЕНО!")
print(f"📁 Проверь файлы в папке: {os.path.abspath(logger.log_dir)}")
```

<img width="983" height="603" alt="image" src="https://github.com/user-attachments/assets/4cc457d6-1c32-451f-8dd8-1a5a8f5c225e" /> <br>

<br>

Statistics tests: <br>

```
# ПРОСТОЙ STATISTICAL TESTS ДЛЯ СРАВНЕНИЯ МОДЕЛЕЙ
print("🎯 ПРОСТОЙ STATISTICAL TESTS (распределение 50/50)")
print("=" * 70)

import json
import time
import numpy as np
from scipy import stats

# Конфигурация
ENDPOINT_NAME = "cat-dog-ab-demo"
TEST_IMAGE = "Bike (31).jpg"

# 1. Собираем данные (30 запросов, быстро)
print("\n📊 СБОР ДАННЫХ (30 запросов)...")
print("-" * 50)

champion_data = {'probabilities': [], 'latencies': [], 'predictions': []}
challenger_data = {'probabilities': [], 'latencies': [], 'predictions': []}

for i in range(30):
    try:
        start_time = time.time()
        
        with open(TEST_IMAGE, "rb") as f:
            image_data = f.read()
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/x-image',
            Body=image_data,
            Accept='application/json'
        )
        
        latency = (time.time() - start_time) * 1000
        result = json.loads(response['Body'].read().decode())
        
        variant = response['ResponseMetadata']['HTTPHeaders'].get(
            'x-amzn-invoked-production-variant', 
            'unknown'
        )
        
        if 'champion' in variant:
            champion_data['probabilities'].append(result['probability'])
            champion_data['latencies'].append(latency)
            champion_data['predictions'].append(result['predicted_label'])
            print(f"  {i+1:2d}: Champion → {result['predicted_label']} ({result['probability']:.1%})")
            
        elif 'challenger' in variant:
            challenger_data['probabilities'].append(result['probability'])
            challenger_data['latencies'].append(latency)
            challenger_data['predictions'].append(result['predicted_label'])
            print(f"  {i+1:2d}: Challenger → {result['predicted_label']} ({result['probability']:.1%})")
            
    except Exception as e:
        print(f"  {i+1:2d}: Ошибка ❌")

print(f"\n✅ Собрано: Champion {len(champion_data['probabilities'])} | Challenger {len(challenger_data['probabilities'])}")

# 2. БАЗОВАЯ СТАТИСТИКА
print("\n📈 БАЗОВАЯ СТАТИСТИКА:")
print("=" * 50)

# Уверенность
champ_probs = np.array(champion_data['probabilities'])
chall_probs = np.array(challenger_data['probabilities'])

print(f"\n🎯 УВЕРЕННОСТЬ:")
print(f"   Champion:  среднее = {np.mean(champ_probs):.4f} (±{np.std(champ_probs):.4f})")
print(f"   Challenger: среднее = {np.mean(chall_probs):.4f} (±{np.std(chall_probs):.4f})")
print(f"   Разница: {np.mean(chall_probs) - np.mean(champ_probs):+.4f}")

# Скорость
champ_lat = np.array(champion_data['latencies'])
chall_lat = np.array(challenger_data['latencies'])

print(f"\n⚡ СКОРОСТЬ:")
print(f"   Champion:  среднее = {np.mean(champ_lat):.1f} мс (±{np.std(champ_lat):.1f})")
print(f"   Challenger: среднее = {np.mean(chall_lat):.1f} мс (±{np.std(chall_lat):.1f})")
print(f"   Разница: {np.mean(chall_lat) - np.mean(champ_lat):+.1f} мс")

# Предсказания
print(f"\n🎯 ПРЕДСКАЗАНИЯ:")
print(f"   Champion всегда говорит: {set(champion_data['predictions'])}")
print(f"   Challenger всегда говорит: {set(challenger_data['predictions'])}")

# 3. ПРОСТОЙ Т-ТЕСТ
print("\n📊 ПРОСТОЙ Т-ТЕСТ:")
print("=" * 50)

# Для уверенности
t_stat_prob, p_value_prob = stats.ttest_ind(champ_probs, chall_probs)
print(f"\n🎯 УВЕРЕННОСТЬ (p-value = {p_value_prob:.6f}):")
if p_value_prob < 0.05:
    if np.mean(chall_probs) > np.mean(champ_probs):
        print("   ✅ Challenger ЗНАЧИТЕЛЬНО УВЕРЕННЕЕ (p < 0.05)")
    else:
        print("   ✅ Champion ЗНАЧИТЕЛЬНО УВЕРЕННЕЕ (p < 0.05)")
else:
    print("   ⚖️  Нет значимой разницы (p ≥ 0.05)")

# Для скорости
t_stat_lat, p_value_lat = stats.ttest_ind(champ_lat, chall_lat)
print(f"\n⚡ СКОРОСТЬ (p-value = {p_value_lat:.6f}):")
if p_value_lat < 0.05:
    if np.mean(chall_lat) < np.mean(champ_lat):
        print("   ✅ Challenger ЗНАЧИТЕЛЬНО БЫСТРЕЕ (p < 0.05)")
    else:
        print("   ✅ Champion ЗНАЧИТЕЛЬНО БЫСТРЕЕ (p < 0.05)")
else:
    print("   ⚖️  Нет значимой разницы (p ≥ 0.05)")

# 4. ПРОСТОЙ ВЕРДИКТ
print("\n" + "=" * 70)
print("🏆 ПРОСТОЙ ВЕРДИКТ:")
print("=" * 70)

score_champ = 0
score_chall = 0

print("\n📋 ОЦЕНКА ПО 3 КРИТЕРИЯМ:")

# 1. Уверенность
if p_value_prob < 0.05:
    if np.mean(chall_probs) > np.mean(champ_probs):
        score_chall += 2
        print("   📈 Challenger +2 (увереннее)")
    else:
        score_champ += 2
        print("   📈 Champion +2 (увереннее)")

# 2. Скорость
if p_value_lat < 0.05:
    if np.mean(chall_lat) < np.mean(champ_lat):
        score_chall += 1
        print("   ⚡ Challenger +1 (быстрее)")
    else:
        score_champ += 1
        print("   ⚡ Champion +1 (быстрее)")

# 3. Консистентность
if len(set(champion_data['predictions'])) == 1:
    score_champ += 0.5
    print("   🎯 Champion +0.5 (стабильные предсказания)")
if len(set(challenger_data['predictions'])) == 1:
    score_chall += 0.5
    print("   🎯 Challenger +0.5 (стабильные предсказания)")

print(f"\n📊 ИТОГОВЫЕ ОЧКИ: Champion {score_champ:.1f} - {score_chall:.1f} Challenger")

if score_chall > score_champ:
    print("\n🎉 ПОБЕДИТЕЛЬ: CHALLENGER")
    print("   Рекомендация: начать Canary deployment")
elif score_champ > score_chall:
    print("\n🎉 ПОБЕДИТЕЛЬ: CHAMPION")  
    print("   Рекомендация: оставить текущую модель")
else:
    print("\n⚖️  НИЧЬЯ")
    print("   Рекомендация: провести больше тестов")



print("\n" + "=" * 70)
print("✅ ПРОСТОЙ STATISTICAL TESTS ЗАВЕРШЁН!")
print("=" * 70)
```

Результат: <br>
```
🎯 ПРОСТОЙ STATISTICAL TESTS (распределение 50/50)
======================================================================

📊 СБОР ДАННЫХ (30 запросов)...
--------------------------------------------------
   1: Challenger → Bike (92.5%)
   2: Champion → Bike (88.2%)
   3: Challenger → Bike (92.5%)
   4: Champion → Bike (88.2%)
   5: Champion → Bike (88.2%)
   6: Champion → Bike (88.2%)
   7: Champion → Bike (88.2%)
   8: Challenger → Bike (92.5%)
   9: Champion → Bike (88.2%)
  10: Champion → Bike (88.2%)
  11: Champion → Bike (88.2%)
  12: Champion → Bike (88.2%)
  13: Champion → Bike (88.2%)
  14: Challenger → Bike (92.5%)
  15: Challenger → Bike (92.5%)
  16: Champion → Bike (88.2%)
  17: Champion → Bike (88.2%)
  18: Champion → Bike (88.2%)
  19: Champion → Bike (88.2%)
  20: Champion → Bike (88.2%)
  21: Champion → Bike (88.2%)
  22: Challenger → Bike (92.5%)
  23: Champion → Bike (88.2%)
  24: Champion → Bike (88.2%)
  25: Challenger → Bike (92.5%)
  26: Challenger → Bike (92.5%)
  27: Champion → Bike (88.2%)
  28: Champion → Bike (88.2%)
  29: Champion → Bike (88.2%)
  30: Champion → Bike (88.2%)

✅ Собрано: Champion 22 | Challenger 8

📈 БАЗОВАЯ СТАТИСТИКА:
==================================================

🎯 УВЕРЕННОСТЬ:
   Champion:  среднее = 0.8821 (±0.0000)
   Challenger: среднее = 0.9250 (±0.0000)
   Разница: +0.0430

⚡ СКОРОСТЬ:
   Champion:  среднее = 1957.6 мс (±97.1)
   Challenger: среднее = 1879.4 мс (±106.7)
   Разница: -78.2 мс

🎯 ПРЕДСКАЗАНИЯ:
   Champion всегда говорит: {'Bike'}
   Challenger всегда говорит: {'Bike'}

📊 ПРОСТОЙ Т-ТЕСТ:
==================================================

🎯 УВЕРЕННОСТЬ (p-value = 0.000000):
   ✅ Challenger ЗНАЧИТЕЛЬНО УВЕРЕННЕЕ (p < 0.05)

⚡ СКОРОСТЬ (p-value = 0.077319):
   ⚖️  Нет значимой разницы (p ≥ 0.05)

======================================================================
🏆 ПРОСТОЙ ВЕРДИКТ:
======================================================================

📋 ОЦЕНКА ПО 3 КРИТЕРИЯМ:
   📈 Challenger +2 (увереннее)
   🎯 Champion +0.5 (стабильные предсказания)
   🎯 Challenger +0.5 (стабильные предсказания)

📊 ИТОГОВЫЕ ОЧКИ: Champion 0.5 - 2.5 Challenger

🎉 ПОБЕДИТЕЛЬ: CHALLENGER
   Рекомендация: начать Canary deployment

======================================================================
✅ ПРОСТОЙ STATISTICAL TESTS ЗАВЕРШЁН!
======================================================================

```

<br>

Canary deployment + Rollback: <br>

```
# CANARY DEPLOYMENT С ОТЛОЖЕННЫМ ROLLBACK (ТОЛЬКО ПОСЛЕ 50/50)
print("🚀 CANARY DEPLOYMENT С ОТЛОЖЕННЫМ ROLLBACK")
print("=" * 70)

import json
import time
import numpy as np
import boto3
from datetime import datetime

class SmartCanary:
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.sagemaker = boto3.client('sagemaker', region_name='us-east-1')
        self.runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.metrics_history = []
        self.warnings_history = []
        
        print(f"🎯 Работаем с эндпоинтом: {endpoint_name}")
        print("🔄 Режим: Rollback только после этапа 50/50")
    
    def get_current_weights(self):
        """Получает текущие веса трафика"""
        try:
            response = self.sagemaker.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            
            print("\n📋 ТЕКУЩЕЕ РАСПРЕДЕЛЕНИЕ ТРАФИКА:")
            print("-" * 40)
            
            variants = []
            for variant in response['ProductionVariants']:
                name = variant['VariantName']
                current_weight = variant.get('CurrentWeight', 0)
                
                variants.append({
                    'name': name,
                    'current_weight': current_weight
                })
                
                print(f"  {name}: {current_weight}%")
            
            return variants
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None
    
    def update_traffic_weights(self, weights):
        """Быстрое обновление весов трафика"""
        print(f"\n⚡ Обновление распределения трафика...")
        
        try:
            desired_weights = []
            
            for variant_name, weight in weights.items():
                desired_weights.append({
                    'VariantName': variant_name,
                    'DesiredWeight': float(weight)
                })
            
            print("📊 Новое распределение:")
            for item in desired_weights:
                print(f"  {item['VariantName']}: {item['DesiredWeight']}%")
            
            response = self.sagemaker.update_endpoint_weights_and_capacities(
                EndpointName=self.endpoint_name,
                DesiredWeightsAndCapacities=desired_weights
            )
            
            print(f"✅ Веса обновлены!")
            print(f"⏳ Ждем 20 секунд для применения изменений...")
            time.sleep(20)
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при обновлении весов: {e}")
            return False
    
    def collect_metrics(self, num_requests=30, stage_name=""):
        """Собирает метрики с эндпоинта"""
        print(f"\n📊 Сбор метрик для {stage_name} ({num_requests} запросов)...")
        
        metrics = {
            'champion': {'count': 0, 'latencies': [], 'confidences': []},
            'challenger': {'count': 0, 'latencies': [], 'confidences': []},
            'stage_name': stage_name,
            'timestamp': datetime.now().isoformat()
        }
        
        for i in range(num_requests):
            try:
                with open("Bike (31).jpg", "rb") as f:
                    image_data = f.read()
                
                start = time.time()
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/x-image',
                    Body=image_data,
                    Accept='application/json'
                )
                
                latency = (time.time() - start) * 1000
                result = json.loads(response['Body'].read().decode())
                
                variant = response['ResponseMetadata']['HTTPHeaders'].get(
                    'x-amzn-invoked-production-variant', 
                    'unknown'
                )
                
                if 'champion' in variant.lower():
                    metrics['champion']['count'] += 1
                    metrics['champion']['latencies'].append(latency)
                    metrics['champion']['confidences'].append(result['probability'])
                    symbol = "🏆"
                elif 'challenger' in variant.lower():
                    metrics['challenger']['count'] += 1
                    metrics['challenger']['latencies'].append(latency)
                    metrics['challenger']['confidences'].append(result['probability'])
                    symbol = "🆕"
                else:
                    symbol = "❓"
                
                print(f"  {i+1:2d}: {symbol} {variant.split('-')[0]} ({latency:.0f}мс, {result['probability']:.1%})")
                time.sleep(0.3)
                
            except Exception as e:
                print(f"  {i+1:2d}: ❌ Ошибка")
        
        # Анализ результатов
        total = metrics['champion']['count'] + metrics['challenger']['count']
        if total > 0:
            champ_percent = (metrics['champion']['count'] / total) * 100
            chall_percent = (metrics['challenger']['count'] / total) * 100
            
            print(f"\n📈 РЕЗУЛЬТАТЫ {stage_name}:")
            print(f"  🏆 Champion: {metrics['champion']['count']} запросов ({champ_percent:.1f}%)")
            print(f"  🆕 Challenger: {metrics['challenger']['count']} запросов ({chall_percent:.1f}%)")
            
            if metrics['champion']['count'] > 0:
                avg_lat = np.mean(metrics['champion']['latencies'])
                avg_conf = np.mean(metrics['champion']['confidences'])
                print(f"    • Задержка: {avg_lat:.0f} мс")
                print(f"    • Уверенность: {avg_conf:.1%}")
            
            if metrics['challenger']['count'] > 0:
                avg_lat = np.mean(metrics['challenger']['latencies'])
                avg_conf = np.mean(metrics['challenger']['confidences'])
                print(f"    • Задержка: {avg_lat:.0f} мс")
                print(f"    • Уверенность: {avg_conf:.1%}")
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_metrics(self, metrics, expected_challenger_percent, stage_name):
        """Анализирует метрики и решает нужен ли rollback"""
        print(f"\n🔍 Анализ метрик для {stage_name}...")
        
        warnings = []
        needs_rollback = False
        
        total = metrics['champion']['count'] + metrics['challenger']['count']
        
        if total == 0:
            warnings.append("Нет данных для анализа")
            return False, warnings
        
        # 1. Проверяем распределение трафика
        actual_chall_percent = (metrics['challenger']['count'] / total) * 100
        
        if actual_chall_percent < expected_challenger_percent * 0.5:
            warnings.append(f"Мало трафика challenger: {actual_chall_percent:.1f}% (ожидалось ~{expected_challenger_percent}%)")
        
        # 2. Проверяем метрики качества (только если есть данные от обоих)
        if metrics['challenger']['count'] > 2 and metrics['champion']['count'] > 2:
            champ_latency = np.mean(metrics['champion']['latencies'])
            chall_latency = np.mean(metrics['challenger']['latencies'])
            champ_conf = np.mean(metrics['champion']['confidences'])
            chall_conf = np.mean(metrics['challenger']['confidences'])
            
            # Проверка задержки (только предупреждение до 50%)
            if chall_latency > champ_latency * 1.5:
                warnings.append(f"Высокая задержка challenger: +{(chall_latency/champ_latency-1)*100:.0f}%")
            
            # Проверка уверенности (только предупреждение до 50%)
            if chall_conf < champ_conf * 0.9:
                warnings.append(f"Низкая уверенность challenger: -{(1-chall_conf/champ_conf)*100:.0f}%")
            
            # КРИТИЧЕСКИЕ ПРОВЕРКИ (вызывают rollback после 50%)
            if expected_challenger_percent >= 50:  # ТОЛЬКО после 50/50 этапа
                if chall_latency > champ_latency * 2.0:  # Очень высокая задержка
                    warnings.append(f"КРИТИЧЕСКОЕ: Задержка challenger > 200% от champion")
                    needs_rollback = True
                
                if chall_conf < champ_conf * 0.8:  # Очень низкая уверенность
                    warnings.append(f"КРИТИЧЕСКОЕ: Уверенность challenger < 80% от champion")
                    needs_rollback = True
                
                if actual_chall_percent < expected_challenger_percent * 0.3:
                    warnings.append(f"КРИТИЧЕСКОЕ: Challenger получает <30% ожидаемого трафика")
                    needs_rollback = True
        
        # Сохраняем предупреждения
        if warnings:
            self.warnings_history.append({
                'stage': stage_name,
                'warnings': warnings,
                'needs_rollback': needs_rollback
            })
        
        # Показываем результаты анализа
        if warnings:
            print("⚠️  ПРЕДУПРЕЖДЕНИЯ:")
            for warning in warnings:
                if "КРИТИЧЕСКОЕ" in warning:
                    print(f"   🔴 {warning}")
                else:
                    print(f"   🟡 {warning}")
        else:
            print("✅ Все метрики в норме")
        
        return needs_rollback, warnings
    
    def perform_rollback(self, champion_var, challenger_var):
        """Выполняет rollback к 100% champion"""
        print(f"\n🔄 ВЫПОЛНЯЕМ ROLLBACK К 100% CHAMPION...")
        
        rollback_weights = {champion_var: 100, challenger_var: 0}
        
        if self.update_traffic_weights(rollback_weights):
            print("✅ Rollback выполнен успешно!")
            print("🏆 100% трафика возвращено Champion модели")
            return True
        else:
            print("❌ Не удалось выполнить rollback!")
            return False
    
    def run_smart_canary(self):
        """Запускает умный canary deployment с отложенным rollback"""
        print("\n" + "=" * 70)
        print("🚀 ЗАПУСК УМНОГО CANARY DEPLOYMENT")
        print("=" * 70)
        print("""
СТРАТЕГИЯ:
• До 50%: Только предупреждения, сбор данных
• После 50%: Критические проверки с rollback
• Rollback срабатывает ТОЛЬКО при серьезных проблемах после 50%
        """)
        
        # Шаг 1: Анализ текущей конфигурации
        print("\n1️⃣  АНАЛИЗ ТЕКУЩЕЙ КОНФИГУРАЦИИ")
        variants = self.get_current_weights()
        
        if not variants:
            print("❌ Не удалось получить конфигурацию")
            return
        
        # Определяем имена вариантов
        champion_var = None
        challenger_var = None
        
        for variant in variants:
            if 'champion' in variant['name'].lower():
                champion_var = variant['name']
            elif 'challenger' in variant['name'].lower():
                challenger_var = variant['name']
        
        if not champion_var or not challenger_var:
            print("❌ Не удалось определить варианты моделей")
            return
        
        print(f"\n   🏆 Champion: {champion_var}")
        print(f"   🆕 Challenger: {challenger_var}")
        
        # Этапы canary deployment
        stages = [
            ("🚀 Canary 10%", {champion_var: 90, challenger_var: 10}),
            ("📈 Canary 25%", {champion_var: 75, challenger_var: 25}),
            ("⚖️  Canary 50%", {champion_var: 50, challenger_var: 50}),
            ("🔥 Canary 75%", {champion_var: 25, challenger_var: 75}),
            ("🎯 Full 100%", {champion_var: 0, challenger_var: 100})
        ]
        
        print("\n2️⃣  ПЛАН DEPLOYMENT:")
        for name, weights in stages:
            champ_w = weights.get(champion_var, 0)
            chall_w = weights.get(challenger_var, 0)
            print(f"   • {name}: Champion {champ_w}% | Challenger {chall_w}%")
        
        input("\n⚠️  Нажмите Enter для начала deployment...")
        
        # Выполняем каждый этап
        for stage_name, target_weights in stages:
            print(f"\n{'='*60}")
            print(f"🎯 ЭТАП: {stage_name}")
            print(f"{'='*60}")
            
            # 1. Обновляем распределение трафика
            print(f"\n⚡ Установка распределения...")
            if not self.update_traffic_weights(target_weights):
                print(f"❌ Ошибка на этапе {stage_name}")
                break
            
            # 2. Собираем метрики
            metrics = self.collect_metrics(
                num_requests=25, 
                stage_name=stage_name
            )
            
            # 3. Анализируем метрики (с разной строгостью)
            expected_chall_percent = target_weights.get(challenger_var, 0)
            needs_rollback, warnings = self.analyze_metrics(
                metrics, expected_chall_percent, stage_name
            )
            
            # 4. Принимаем решение о rollback
            if needs_rollback:
                print(f"\n❌ КРИТИЧЕСКИЕ ПРОБЛЕМЫ НА ЭТАПЕ '{stage_name}'!")
                print("🔄 Выполняем автоматический rollback...")
                
                if self.perform_rollback(champion_var, challenger_var):
                    print(f"\n📋 ИТОГ: Deployment прерван на этапе {stage_name}")
                    self.show_summary()
                    return
                else:
                    print("❌ Не удалось выполнить rollback!")
                    break
            elif warnings:
                print(f"\n⚠️  Есть предупреждения, но продолжаем deployment...")
            else:
                print(f"\n✅ Этап '{stage_name}' успешно завершен!")
            
            # Пауза между этапами
            if stage_name != stages[-1][0]:
                print(f"\n⏸️  Пауза 45 секунд перед следующим этапом...")
                time.sleep(45)
        
        print("\n" + "=" * 70)
        print("🎉 CANARY DEPLOYMENT ЗАВЕРШЁН УСПЕШНО!")
        print("=" * 70)
        print("✅ Challenger модель теперь обрабатывает 100% трафика")
        print("✅ Все этапы пройдены")
        print("✅ Производственная система обновлена")
        
        self.show_summary()
    
    def show_summary(self):
        """Показывает итоговую статистику"""
        if not self.metrics_history:
            return
        
        print("\n" + "=" * 70)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 70)
        
        print("\n📈 ИСТОРИЯ МЕТРИК:")
        for metrics in self.metrics_history:
            total = metrics['champion']['count'] + metrics['challenger']['count']
            if total > 0:
                chall_percent = (metrics['challenger']['count'] / total) * 100
                print(f"  {metrics['stage_name']}: Challenger {chall_percent:.1f}% трафика")
        
        if self.warnings_history:
            print("\n⚠️  ИСТОРИЯ ПРЕДУПРЕЖДЕНИЙ:")
            for warning_entry in self.warnings_history:
                print(f"\n  {warning_entry['stage']}:")
                for warning in warning_entry['warnings']:
                    if "КРИТИЧЕСКОЕ" in warning:
                        print(f"    🔴 {warning}")
                    else:
                        print(f"    🟡 {warning}")

# ЗАПУСК УМНОГО CANARY DEPLOYMENT
def main():
    print("🧠 УМНЫЙ CANARY DEPLOYMENT С ОТЛОЖЕННЫМ ROLLBACK")
    print("=" * 70)
    print("""
Ключевые особенности:
1. 🎯 Rollback только ПОСЛЕ этапа 50/50
2. 📊 До 50%: только предупреждения и сбор данных
3. 🔥 После 50%: критические проверки с rollback
4. 📈 Постепенное увеличение: 10% → 25% → 50% → 75% → 100%
5. ⏱️  Более реалистичный подход для production
    """)
    
    ENDPOINT_NAME = "cat-dog-ab-demo"
    
    # Создаем canary deployer
    canary = SmartCanary(ENDPOINT_NAME)
    
    # Запускаем deployment
    canary.run_smart_canary()

# АЛЬТЕРНАТИВНАЯ ВЕРСИЯ: ПРОСТОЙ ЗАПУСК
def simple_run():
    """Простой запуск без меню"""
    ENDPOINT_NAME = "cat-dog-ab-demo"
    canary = SmartCanary(ENDPOINT_NAME)
    canary.run_smart_canary()

# Выбор режима
if __name__ == "__main__":
    print("Выберите режим запуска:")
    print("1. 🧠 Умный canary deployment (рекомендуется)")
    print("2. 🚀 Простой запуск")
    
    choice = input("\nВаш выбор (1-2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        simple_run()
    else:
        print("❌ Неверный выбор, запускаю умную версию...")
        main()
```

ИТОГ: <br>

(Я отложил rollback до распределения 50/50 т.к. при меньшем распределении chellenger получал мало трафика и соответсвенно оключался) <br>

```
🚀 CANARY DEPLOYMENT С ОТЛОЖЕННЫМ ROLLBACK
======================================================================
Выберите режим запуска:
1. 🧠 Умный canary deployment (рекомендуется)
2. 🚀 Простой запуск

Ваш выбор (1-2):  1
🧠 УМНЫЙ CANARY DEPLOYMENT С ОТЛОЖЕННЫМ ROLLBACK
======================================================================

Ключевые особенности:
1. 🎯 Rollback только ПОСЛЕ этапа 50/50
2. 📊 До 50%: только предупреждения и сбор данных
3. 🔥 После 50%: критические проверки с rollback
4. 📈 Постепенное увеличение: 10% → 25% → 50% → 75% → 100%
5. ⏱️  Более реалистичный подход для production
    
🎯 Работаем с эндпоинтом: cat-dog-ab-demo
🔄 Режим: Rollback только после этапа 50/50

======================================================================
🚀 ЗАПУСК УМНОГО CANARY DEPLOYMENT
======================================================================

СТРАТЕГИЯ:
• До 50%: Только предупреждения, сбор данных
• После 50%: Критические проверки с rollback
• Rollback срабатывает ТОЛЬКО при серьезных проблемах после 50%
        

1️⃣  АНАЛИЗ ТЕКУЩЕЙ КОНФИГУРАЦИИ

📋 ТЕКУЩЕЕ РАСПРЕДЕЛЕНИЕ ТРАФИКА:
----------------------------------------
  champion-variant: 100.0%
  challenger-variant: 0.0%

   🏆 Champion: champion-variant
   🆕 Challenger: challenger-variant

2️⃣  ПЛАН DEPLOYMENT:
   • 🚀 Canary 10%: Champion 90% | Challenger 10%
   • 📈 Canary 25%: Champion 75% | Challenger 25%
   • ⚖️  Canary 50%: Champion 50% | Challenger 50%
   • 🔥 Canary 75%: Champion 25% | Challenger 75%
   • 🎯 Full 100%: Champion 0% | Challenger 100%

⚠️  Нажмите Enter для начала deployment... 

============================================================
🎯 ЭТАП: 🚀 Canary 10%
============================================================

⚡ Установка распределения...

⚡ Обновление распределения трафика...
📊 Новое распределение:
  champion-variant: 90.0%
  challenger-variant: 10.0%
✅ Веса обновлены!
⏳ Ждем 20 секунд для применения изменений...

📊 Сбор метрик для 🚀 Canary 10% (25 запросов)...
   1: 🏆 champion (1877мс, 88.2%)
   2: 🏆 champion (1764мс, 88.2%)
   3: 🏆 champion (1755мс, 88.2%)
   4: 🏆 champion (1815мс, 88.2%)
   5: 🏆 champion (1785мс, 88.2%)
   6: 🆕 challenger (2373мс, 92.5%)
   7: 🏆 champion (1756мс, 88.2%)
   8: 🏆 champion (1774мс, 88.2%)
   9: 🏆 champion (1746мс, 88.2%)
  10: 🏆 champion (1902мс, 88.2%)
  11: 🏆 champion (1927мс, 88.2%)
  12: 🏆 champion (1758мс, 88.2%)
  13: 🏆 champion (1754мс, 88.2%)
  14: 🏆 champion (1779мс, 88.2%)
  15: 🏆 champion (1783мс, 88.2%)
  16: 🏆 champion (1777мс, 88.2%)
  17: 🏆 champion (1746мс, 88.2%)
  18: 🏆 champion (1760мс, 88.2%)
  19: 🏆 champion (1789мс, 88.2%)
  20: 🏆 champion (1752мс, 88.2%)
  21: 🏆 champion (1762мс, 88.2%)
  22: 🏆 champion (1790мс, 88.2%)
  23: 🏆 champion (1771мс, 88.2%)
  24: 🏆 champion (1748мс, 88.2%)
  25: 🏆 champion (1754мс, 88.2%)

📈 РЕЗУЛЬТАТЫ 🚀 Canary 10%:
  🏆 Champion: 24 запросов (96.0%)
  🆕 Challenger: 1 запросов (4.0%)
    • Задержка: 1784 мс
    • Уверенность: 88.2%
    • Задержка: 2373 мс
    • Уверенность: 92.5%

🔍 Анализ метрик для 🚀 Canary 10%...
⚠️  ПРЕДУПРЕЖДЕНИЯ:
   🟡 Мало трафика challenger: 4.0% (ожидалось ~10%)

⚠️  Есть предупреждения, но продолжаем deployment...

⏸️  Пауза 45 секунд перед следующим этапом...

============================================================
🎯 ЭТАП: 📈 Canary 25%
============================================================

⚡ Установка распределения...

⚡ Обновление распределения трафика...
📊 Новое распределение:
  champion-variant: 75.0%
  challenger-variant: 25.0%
✅ Веса обновлены!
⏳ Ждем 20 секунд для применения изменений...

📊 Сбор метрик для 📈 Canary 25% (25 запросов)...
   1: 🆕 challenger (2102мс, 92.5%)
   2: 🏆 champion (1873мс, 88.2%)
   3: 🏆 champion (1757мс, 88.2%)
   4: 🆕 challenger (2157мс, 92.5%)
   5: 🏆 champion (1816мс, 88.2%)
   6: 🆕 challenger (2173мс, 92.5%)
   7: 🏆 champion (1755мс, 88.2%)
   8: 🏆 champion (1766мс, 88.2%)
   9: 🏆 champion (1799мс, 88.2%)
  10: 🏆 champion (1752мс, 88.2%)
  11: 🆕 challenger (1982мс, 92.5%)
  12: 🏆 champion (1794мс, 88.2%)
  13: 🆕 challenger (1988мс, 92.5%)
  14: 🏆 champion (1756мс, 88.2%)
  15: 🏆 champion (1750мс, 88.2%)
  16: 🏆 champion (1774мс, 88.2%)
  17: 🏆 champion (1758мс, 88.2%)
  18: 🏆 champion (1905мс, 88.2%)
  19: 🆕 challenger (1951мс, 92.5%)
  20: 🏆 champion (1908мс, 88.2%)
  21: 🏆 champion (1754мс, 88.2%)
  22: 🆕 challenger (1939мс, 92.5%)
  23: 🏆 champion (1781мс, 88.2%)
  24: 🏆 champion (1835мс, 88.2%)
  25: 🏆 champion (1769мс, 88.2%)

📈 РЕЗУЛЬТАТЫ 📈 Canary 25%:
  🏆 Champion: 18 запросов (72.0%)
  🆕 Challenger: 7 запросов (28.0%)
    • Задержка: 1795 мс
    • Уверенность: 88.2%
    • Задержка: 2042 мс
    • Уверенность: 92.5%

🔍 Анализ метрик для 📈 Canary 25%...
✅ Все метрики в норме

✅ Этап '📈 Canary 25%' успешно завершен!

⏸️  Пауза 45 секунд перед следующим этапом...

============================================================
🎯 ЭТАП: ⚖️  Canary 50%
============================================================

⚡ Установка распределения...

⚡ Обновление распределения трафика...
📊 Новое распределение:
  champion-variant: 50.0%
  challenger-variant: 50.0%
✅ Веса обновлены!
⏳ Ждем 20 секунд для применения изменений...

📊 Сбор метрик для ⚖️  Canary 50% (25 запросов)...
   1: 🏆 champion (1823мс, 88.2%)
   2: 🆕 challenger (1939мс, 92.5%)
   3: 🏆 champion (1831мс, 88.2%)
   4: 🆕 challenger (1952мс, 92.5%)
   5: 🏆 champion (1810мс, 88.2%)
   6: 🏆 champion (1756мс, 88.2%)
   7: 🆕 challenger (1979мс, 92.5%)
   8: 🏆 champion (1752мс, 88.2%)
   9: 🆕 challenger (1889мс, 92.5%)
  10: 🏆 champion (1758мс, 88.2%)
  11: 🆕 challenger (1899мс, 92.5%)
  12: 🆕 challenger (1861мс, 92.5%)
  13: 🆕 challenger (1895мс, 92.5%)
  14: 🏆 champion (1751мс, 88.2%)
  15: 🏆 champion (1754мс, 88.2%)
  16: 🏆 champion (1756мс, 88.2%)
  17: 🏆 champion (1792мс, 88.2%)
  18: 🆕 challenger (1836мс, 92.5%)
  19: 🏆 champion (1756мс, 88.2%)
  20: 🏆 champion (1760мс, 88.2%)
  21: 🏆 champion (1792мс, 88.2%)
  22: 🏆 champion (1749мс, 88.2%)
  23: 🏆 champion (1751мс, 88.2%)
  24: 🏆 champion (1765мс, 88.2%)
  25: 🏆 champion (1751мс, 88.2%)

📈 РЕЗУЛЬТАТЫ ⚖️  Canary 50%:
  🏆 Champion: 17 запросов (68.0%)
  🆕 Challenger: 8 запросов (32.0%)
    • Задержка: 1771 мс
    • Уверенность: 88.2%
    • Задержка: 1906 мс
    • Уверенность: 92.5%

🔍 Анализ метрик для ⚖️  Canary 50%...
✅ Все метрики в норме

✅ Этап '⚖️  Canary 50%' успешно завершен!

⏸️  Пауза 45 секунд перед следующим этапом...

============================================================
🎯 ЭТАП: 🔥 Canary 75%
============================================================

⚡ Установка распределения...

⚡ Обновление распределения трафика...
📊 Новое распределение:
  champion-variant: 25.0%
  challenger-variant: 75.0%
✅ Веса обновлены!
⏳ Ждем 20 секунд для применения изменений...

📊 Сбор метрик для 🔥 Canary 75% (25 запросов)...
   1: 🆕 challenger (1990мс, 92.5%)
   2: 🆕 challenger (1912мс, 92.5%)
   3: 🆕 challenger (1921мс, 92.5%)
   4: 🏆 champion (1762мс, 88.2%)
   5: 🏆 champion (1771мс, 88.2%)
   6: 🆕 challenger (1951мс, 92.5%)
   7: 🆕 challenger (1954мс, 92.5%)
   8: 🆕 challenger (1960мс, 92.5%)
   9: 🆕 challenger (1922мс, 92.5%)
  10: 🆕 challenger (1900мс, 92.5%)
  11: 🆕 challenger (1869мс, 92.5%)
  12: 🆕 challenger (1916мс, 92.5%)
  13: 🆕 challenger (1925мс, 92.5%)
  14: 🆕 challenger (1914мс, 92.5%)
  15: 🆕 challenger (1943мс, 92.5%)
  16: 🏆 champion (1752мс, 88.2%)
  17: 🆕 challenger (1959мс, 92.5%)
  18: 🆕 challenger (1928мс, 92.5%)
  19: 🆕 challenger (1955мс, 92.5%)
  20: 🆕 challenger (1874мс, 92.5%)
  21: 🏆 champion (1756мс, 88.2%)
  22: 🆕 challenger (1909мс, 92.5%)
  23: 🆕 challenger (1887мс, 92.5%)
  24: 🆕 challenger (1939мс, 92.5%)
  25: 🆕 challenger (1930мс, 92.5%)

📈 РЕЗУЛЬТАТЫ 🔥 Canary 75%:
  🏆 Champion: 4 запросов (16.0%)
  🆕 Challenger: 21 запросов (84.0%)
    • Задержка: 1760 мс
    • Уверенность: 88.2%
    • Задержка: 1927 мс
    • Уверенность: 92.5%

🔍 Анализ метрик для 🔥 Canary 75%...
✅ Все метрики в норме

✅ Этап '🔥 Canary 75%' успешно завершен!

⏸️  Пауза 45 секунд перед следующим этапом...

============================================================
🎯 ЭТАП: 🎯 Full 100%
============================================================

⚡ Установка распределения...

⚡ Обновление распределения трафика...
📊 Новое распределение:
  champion-variant: 0.0%
  challenger-variant: 100.0%
✅ Веса обновлены!
⏳ Ждем 20 секунд для применения изменений...

📊 Сбор метрик для 🎯 Full 100% (25 запросов)...
   1: 🆕 challenger (1951мс, 92.5%)
   2: 🆕 challenger (2029мс, 92.5%)
   3: 🆕 challenger (2081мс, 92.5%)
   4: 🆕 challenger (1894мс, 92.5%)
   5: 🆕 challenger (1972мс, 92.5%)
   6: 🆕 challenger (1883мс, 92.5%)
   7: 🆕 challenger (1914мс, 92.5%)
   8: 🆕 challenger (1969мс, 92.5%)
   9: 🆕 challenger (1883мс, 92.5%)
  10: 🆕 challenger (1886мс, 92.5%)
  11: 🆕 challenger (1916мс, 92.5%)
  12: 🆕 challenger (1917мс, 92.5%)
  13: 🆕 challenger (1916мс, 92.5%)
  14: 🆕 challenger (1956мс, 92.5%)
  15: 🆕 challenger (1924мс, 92.5%)
  16: 🆕 challenger (1926мс, 92.5%)
  17: 🆕 challenger (1894мс, 92.5%)
  18: 🆕 challenger (1902мс, 92.5%)
  19: 🆕 challenger (1905мс, 92.5%)
  20: 🆕 challenger (1918мс, 92.5%)
  21: 🆕 challenger (1911мс, 92.5%)
  22: 🆕 challenger (1919мс, 92.5%)
  23: 🆕 challenger (1939мс, 92.5%)
  24: 🆕 challenger (1898мс, 92.5%)
  25: 🆕 challenger (1895мс, 92.5%)

📈 РЕЗУЛЬТАТЫ 🎯 Full 100%:
  🏆 Champion: 0 запросов (0.0%)
  🆕 Challenger: 25 запросов (100.0%)
    • Задержка: 1928 мс
    • Уверенность: 92.5%

🔍 Анализ метрик для 🎯 Full 100%...
✅ Все метрики в норме

✅ Этап '🎯 Full 100%' успешно завершен!

======================================================================
🎉 CANARY DEPLOYMENT ЗАВЕРШЁН УСПЕШНО!
======================================================================
✅ Challenger модель теперь обрабатывает 100% трафика
✅ Все этапы пройдены
✅ Производственная система обновлена

======================================================================
📊 ИТОГОВАЯ СТАТИСТИКА
======================================================================

📈 ИСТОРИЯ МЕТРИК:
  🚀 Canary 10%: Challenger 4.0% трафика
  📈 Canary 25%: Challenger 28.0% трафика
  ⚖️  Canary 50%: Challenger 32.0% трафика
  🔥 Canary 75%: Challenger 84.0% трафика
  🎯 Full 100%: Challenger 100.0% трафика

⚠️  ИСТОРИЯ ПРЕДУПРЕЖДЕНИЙ:

  🚀 Canary 10%:
    🟡 Мало трафика challenger: 4.0% (ожидалось ~10%)

```



