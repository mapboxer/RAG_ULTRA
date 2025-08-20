#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для оптимизированной RAG системы
Включает тесты поисковика, реранкера, контекст-билдера и графового анализатора
"""

import time
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

try:
    from retrieval.searcher import SemanticSearcher
    from retrieval.universal_searcher import UniversalSearcher, UniversalSearchConfig, Weights, Gating
    from retrieval.reranker import HierarchicalReranker, RerankerConfig
    from retrieval.context_builder import HierarchicalContextBuilder, ContextConfig
    from graph_builder import HierarchicalGraphBuilder
    print("✓ Все модули успешно импортированы")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)


class RAGSystemTester:
    """Тестер для всех компонентов RAG системы"""

    def __init__(self):
        self.base_searcher = None
        self.reranker = None
        self.universal_searcher = None
        self.context_builder = None
        self.graph_builder = None
        self.test_results = {}

    def test_initialization(self):
        """Тест инициализации всех компонентов"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 1: Инициализация компонентов")
        print("="*60)

        try:
            # 1. Базовый поисковик
            print("1.1 Инициализация базового поисковика...")
            start_time = time.time()
            self.base_searcher = SemanticSearcher()
            init_time = time.time() - start_time
            print(f"   ✓ Базовый поисковик загружен за {init_time:.2f}с")

            # 2. Реранкер
            print("1.2 Инициализация реранкера...")
            reranker_config = RerankerConfig(
                kind="hybrid",
                model_path="models/reranker_ru",
                batch_size=8,
                enable_cache=True,
                cache_size=1000,
                adaptive_batching=True,
                hierarchy_aware=True
            )

            start_time = time.time()
            self.reranker = HierarchicalReranker(
                reranker_config,
                embed_encode_fn=lambda t: self.base_searcher.model.encode(
                    t, convert_to_numpy=True, normalize_embeddings=True
                ),
                tokenizer=self.base_searcher.tokenizer
            )
            init_time = time.time() - start_time
            print(f"   ✓ Реранкер инициализирован за {init_time:.2f}с")

            # 3. Универсальный поисковик
            print("1.3 Настройка универсального поисковика...")
            search_config = UniversalSearchConfig(
                semantic_top_k=100,
                top_k_docs=5,
                max_chunks_per_doc=4,
                enable_cache=True,
                cache_size=500
            )

            start_time = time.time()
            self.universal_searcher = UniversalSearcher(
                base_searcher=self.base_searcher,
                cfg=search_config,
                final_reranker=self.reranker
            )
            init_time = time.time() - start_time
            print(f"   ✓ Универсальный поисковик настроен за {init_time:.2f}с")

            # 4. Контекст-билдер
            print("1.4 Инициализация контекст-билдера...")
            ctx_config = ContextConfig(
                token_budget=2000,
                per_doc_max_chunks=3,
                preserve_hierarchy=True,
                group_by_sections=True
            )

            start_time = time.time()
            self.context_builder = HierarchicalContextBuilder(
                ctx_config,
                tokenizer=self.base_searcher.tokenizer
            )
            init_time = time.time() - start_time
            print(f"   ✓ Контекст-билдер инициализирован за {init_time:.2f}с")

            # 5. Графовый анализатор
            print("1.5 Инициализация графового анализатора...")
            start_time = time.time()
            self.graph_builder = HierarchicalGraphBuilder()
            init_time = time.time() - start_time
            print(
                f"   ✓ Графовый анализатор инициализирован за {init_time:.2f}с")

            self.test_results['initialization'] = True
            print("✅ Все компоненты успешно инициализированы!")

        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            traceback.print_exc()
            self.test_results['initialization'] = False
            return False

        return True

    def test_basic_search(self):
        """Тест базового поиска"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 2: Базовый поиск")
        print("="*60)

        if not self.base_searcher:
            print("❌ Базовый поисковик не инициализирован")
            return False

        test_queries = [
            "условия поставки нерудных материалов железнодорожным транспортом",
            "ответственность покупателя при поставке",
            "общие условия договора поставки",
            "порядок согласования заказов"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n2.{i} Поиск: '{query[:50]}...'")

            try:
                start_time = time.time()
                results = self.base_searcher.search_raw(query, top_k=5)
                search_time = time.time() - start_time

                print(f"   Время поиска: {search_time:.3f}с")
                print(f"   Найдено документов: {len(results.results)}")
                print(f"   Всего кандидатов: {results.total_candidates}")

                if results.results:
                    top_result = results.results[0]
                    print(
                        f"   Лучший результат: {top_result.category} | {top_result.doc_path}")
                    print(f"   Лучший скор: {top_result.best_score:.3f}")

                self.test_results[f'basic_search_{i}'] = True

            except Exception as e:
                print(f"   ❌ Ошибка поиска: {e}")
                self.test_results[f'basic_search_{i}'] = False

        return True

    def test_universal_search(self):
        """Тест универсального поиска"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 3: Универсальный поиск")
        print("="*60)

        if not self.universal_searcher:
            print("❌ Универсальный поисковик не инициализирован")
            return False

        test_query = "условия поставки нерудных материалов железнодорожным транспортом"
        print(f"3.1 Поиск: '{test_query[:50]}...'")

        try:
            # Первый поиск
            start_time = time.time()
            results1 = self.universal_searcher.search(test_query)
            search_time1 = time.time() - start_time

            print(f"   Первый поиск: {search_time1:.3f}с")
            print(f"   Найдено документов: {len(results1.results)}")

            # Второй поиск (должен быть быстрее благодаря кэшу)
            start_time = time.time()
            results2 = self.universal_searcher.search(test_query)
            search_time2 = time.time() - start_time

            print(f"   Повторный поиск: {search_time2:.3f}с")
            print(f"   Ускорение: {search_time1/search_time2:.1f}x")

            # Статистика кэша
            cache_stats = self.universal_searcher.get_cache_stats()
            print(f"   Статистика кэша: {cache_stats}")

            self.test_results['universal_search'] = True

        except Exception as e:
            print(f"   ❌ Ошибка универсального поиска: {e}")
            traceback.print_exc()
            self.test_results['universal_search'] = False
            return False

        return True

    def test_reranker(self):
        """Тест реранкера"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 4: Реранкер")
        print("="*60)

        if not self.reranker or not self.base_searcher:
            print("❌ Реранкер или базовый поисковик не инициализированы")
            return False

        test_query = "условия поставки нерудных материалов железнодорожным транспортом"
        print(f"4.1 Тест реранкера для запроса: '{test_query[:50]}...'")

        try:
            # Получаем базовые результаты
            base_results = self.base_searcher.search_raw(test_query, top_k=10)

            if not base_results.results:
                print("   ❌ Нет результатов для реранкинга")
                return False

            print(f"   Базовых результатов: {len(base_results.results)}")

            # Применяем реранкер
            start_time = time.time()
            reranked = self.reranker.rerank(
                test_query,
                base_results,
                top_k_docs=5,
                max_chunks_per_doc=3
            )
            rerank_time = time.time() - start_time

            print(f"   Время реранкинга: {rerank_time:.3f}с")
            print(f"   Реранкнутых документов: {len(reranked)}")

            # Анализируем результаты
            for i, doc in enumerate(reranked[:3], 1):
                print(f"   {i}. {doc.category} | {doc.doc_path}")
                print(f"      Скор: {doc.best_score:.3f}")
                print(f"      Иерархия: {doc.hierarchy_score:.3f}")
                print(f"      Связность: {doc.structural_coherence:.3f}")
                print(f"      Хитов: {len(doc.hits)}")

            # Статистика кэша реранкера
            cache_stats = self.reranker.get_cache_stats()
            print(f"   Статистика кэша реранкера: {cache_stats}")

            self.test_results['reranker'] = True

        except Exception as e:
            print(f"   ❌ Ошибка реранкера: {e}")
            traceback.print_exc()
            self.test_results['reranker'] = False
            return False

        return True

    def test_context_builder(self):
        """Тест контекст-билдера"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 5: Контекст-билдер")
        print("="*60)

        if not self.context_builder or not self.reranker:
            print("❌ Контекст-билдер или реранкер не инициализированы")
            return False

        test_query = "условия поставки нерудных материалов железнодорожным транспортом"
        print(f"5.1 Тест контекст-билдера для запроса: '{test_query[:50]}...'")

        try:
            # Получаем реранкнутые результаты
            base_results = self.base_searcher.search_raw(test_query, top_k=8)
            reranked = self.reranker.rerank(
                test_query, base_results, top_k_docs=4, max_chunks_per_doc=3)

            if not reranked:
                print("   ❌ Нет реранкнутых результатов для контекста")
                return False

            # Строим контекст
            start_time = time.time()
            context_pack = self.context_builder.build(reranked)
            build_time = time.time() - start_time

            print(f"   Время построения контекста: {build_time:.3f}с")
            print(f"   Количество элементов: {len(context_pack.items)}")
            print(f"   Длина промпта: {len(context_pack.prompt)} символов")
            print(f"   Количество цитат: {len(context_pack.citations)}")

            # Анализируем качество
            quality_metrics = self.context_builder.get_quality_metrics(
                context_pack)
            print(f"   Метрики качества:")
            for key, value in quality_metrics.items():
                print(f"     {key}: {value}")

            # Структура контекста
            print(f"   Структура: {context_pack.hierarchy_summary}")

            # Покрытие
            print(
                f"   Покрытие категорий: {context_pack.coverage_stats.get('category_coverage', 0)}")
            print(
                f"   Покрытие разделов: {context_pack.coverage_stats.get('section_coverage', 0)}")

            self.test_results['context_builder'] = True

        except Exception as e:
            print(f"   ❌ Ошибка контекст-билдера: {e}")
            traceback.print_exc()
            self.test_results['context_builder'] = False
            return False

        return True

    def test_graph_analyzer(self):
        """Тест графового анализатора"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 6: Графовый анализатор")
        print("="*60)

        if not self.graph_builder:
            print("❌ Графовый анализатор не инициализирован")
            return False

        print("6.1 Тест анализа иерархии документов")

        try:
            # Анализируем структуру найденных документов
            test_query = "условия поставки нерудных материалов железнодорожным транспортом"
            base_results = self.base_searcher.search_raw(test_query, top_k=5)

            if not base_results.results:
                print("   ❌ Нет результатов для анализа")
                return False

            # Анализируем заголовки
            section_analysis = {}
            for doc in base_results.results:
                if doc.hits:
                    headings = []
                    for hit in doc.hits:
                        if hit.heading_path:
                            headings.extend(hit.heading_path)

                    # Группируем по уровням
                    levels = {}
                    for heading in headings:
                        level = heading.count('.') + 1
                        if level not in levels:
                            levels[level] = []
                        levels[level].append(heading)

                    section_analysis[doc.doc_path] = {
                        'total_headings': len(headings),
                        'levels': levels,
                        'max_level': max(levels.keys()) if levels else 0
                    }

            print(f"   Проанализировано документов: {len(section_analysis)}")

            # Выводим анализ
            for doc_path, analysis in section_analysis.items():
                print(f"   Документ: {doc_path}")
                print(f"     Всего заголовков: {analysis['total_headings']}")
                print(f"     Максимальный уровень: {analysis['max_level']}")
                for level, headings in analysis['levels'].items():
                    print(f"     Уровень {level}: {len(headings)} заголовков")

            # Тестируем поиск релевантных разделов
            print("\n6.2 Тест поиска релевантных разделов")
            relevant_sections = self.graph_builder.get_relevant_sections(
                test_query, top_k=3)
            print(f"   Найдено релевантных разделов: {len(relevant_sections)}")

            for section_id, score in relevant_sections:
                print(f"     {section_id}: {score:.3f}")

            self.test_results['graph_analyzer'] = True

        except Exception as e:
            print(f"   ❌ Ошибка графового анализатора: {e}")
            traceback.print_exc()
            self.test_results['graph_analyzer'] = False
            return False

        return True

    def test_performance_optimizations(self):
        """Тест оптимизаций производительности"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 7: Оптимизации производительности")
        print("="*60)

        if not all([self.base_searcher, self.universal_searcher, self.reranker]):
            print("❌ Не все компоненты инициализированы")
            return False

        test_query = "условия поставки нерудных материалов железнодорожным транспортом"
        print(
            f"7.1 Тест производительности для запроса: '{test_query[:50]}...'")

        try:
            # Тест базового поиска
            print("   Тест базового поиска...")
            start_time = time.time()
            base_results = self.base_searcher.search_raw(test_query, top_k=10)
            base_time = time.time() - start_time

            # Тест универсального поиска
            print("   Тест универсального поиска...")
            start_time = time.time()
            universal_results = self.universal_searcher.search(test_query)
            universal_time = time.time() - start_time

            # Тест реранкера
            print("   Тест реранкера...")
            start_time = time.time()
            reranked = self.reranker.rerank(
                test_query, base_results, top_k_docs=5, max_chunks_per_doc=3)
            rerank_time = time.time() - start_time

            # Выводим результаты
            print(f"   Базовый поиск: {base_time:.3f}с")
            print(f"   Универсальный поиск: {universal_time:.3f}с")
            print(f"   Реранкер: {rerank_time:.3f}с")

            # Анализируем эффективность
            if base_time > 0:
                universal_overhead = (
                    universal_time - base_time) / base_time * 100
                print(
                    f"   Накладные расходы универсального поиска: {universal_overhead:.1f}%")

            # Статистика кэша
            print("\n   Статистика кэша:")
            base_stats = self.base_searcher.get_cache_stats()
            universal_stats = self.universal_searcher.get_cache_stats()
            reranker_stats = self.reranker.get_cache_stats()

            print(f"     Базовый поиск: {base_stats}")
            print(f"     Универсальный поиск: {universal_stats}")
            print(f"     Реранкер: {reranker_stats}")

            self.test_results['performance_optimizations'] = True

        except Exception as e:
            print(f"   ❌ Ошибка теста производительности: {e}")
            traceback.print_exc()
            self.test_results['performance_optimizations'] = False
            return False

        return True

    def test_cache_management(self):
        """Тест управления кэшем"""
        print("\n" + "="*60)
        print("🧪 ТЕСТ 8: Управление кэшем")
        print("="*60)

        if not all([self.base_searcher, self.universal_searcher, self.reranker]):
            print("❌ Не все компоненты инициализированы")
            return False

        try:
            test_query = "условия поставки нерудных материалов железнодорожным транспортом"

            # Первый поиск (заполняет кэш)
            print("8.1 Первый поиск (заполнение кэша)...")
            start_time = time.time()
            results1 = self.universal_searcher.search(test_query)
            time1 = time.time() - start_time

            # Второй поиск (из кэша)
            print("8.2 Повторный поиск (из кэша)...")
            start_time = time.time()
            results2 = self.universal_searcher.search(test_query)
            time2 = time.time() - start_time

            print(f"   Первый поиск: {time1:.3f}с")
            print(f"   Повторный поиск: {time2:.3f}с")
            print(f"   Ускорение: {time1/time2:.1f}x")

            # Статистика до очистки
            print("\n8.3 Статистика кэша до очистки:")
            base_stats = self.base_searcher.get_cache_stats()
            universal_stats = self.universal_searcher.get_cache_stats()
            reranker_stats = self.reranker.get_cache_stats()

            print(f"   Базовый поиск: {base_stats}")
            print(f"   Универсальный поиск: {universal_stats}")
            print(f"   Реранкер: {reranker_stats}")

            # Очистка кэша
            print("\n8.4 Очистка кэша...")
            self.base_searcher.clear_cache()
            self.universal_searcher.clear_cache()
            self.reranker.clear_cache()

            # Статистика после очистки
            print("8.5 Статистика кэша после очистки:")
            base_stats = self.base_searcher.get_cache_stats()
            universal_stats = self.universal_searcher.get_cache_stats()
            reranker_stats = self.reranker.get_cache_stats()

            print(f"   Базовый поиск: {base_stats}")
            print(f"   Универсальный поиск: {universal_stats}")
            print(f"   Реранкер: {reranker_stats}")

            self.test_results['cache_management'] = True

        except Exception as e:
            print(f"   ❌ Ошибка управления кэшем: {e}")
            traceback.print_exc()
            self.test_results['cache_management'] = False
            return False

        return True

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("🚀 ЗАПУСК ТЕСТОВ RAG СИСТЕМЫ")
        print("="*60)

        tests = [
            ("Инициализация", self.test_initialization),
            ("Базовый поиск", self.test_basic_search),
            ("Универсальный поиск", self.test_universal_search),
            ("Реранкер", self.test_reranker),
            ("Контекст-билдер", self.test_context_builder),
            ("Графовый анализатор", self.test_graph_analyzer),
            ("Оптимизации производительности", self.test_performance_optimizations),
            ("Управление кэшем", self.test_cache_management)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name}: ПРОЙДЕН")
                else:
                    print(f"❌ {test_name}: ПРОВАЛЕН")
            except Exception as e:
                print(f"❌ {test_name}: ОШИБКА - {e}")
                traceback.print_exc()

        # Итоговый отчет
        print("\n" + "="*60)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("="*60)
        print(f"Всего тестов: {total}")
        print(f"Пройдено: {passed}")
        print(f"Провалено: {total - passed}")
        print(f"Процент успеха: {passed/total*100:.1f}%")

        # Детальные результаты
        print("\nДетальные результаты:")
        for key, value in self.test_results.items():
            status = "✅ ПРОЙДЕН" if value else "❌ ПРОВАЛЕН"
            print(f"  {key}: {status}")

        return passed == total


def main():
    """Основная функция"""
    print("🧪 Тестирование оптимизированной RAG системы")
    print("="*60)

    try:
        tester = RAGSystemTester()
        success = tester.run_all_tests()

        if success:
            print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            return 0
        else:
            print("\n⚠️  НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
            return 1

    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
