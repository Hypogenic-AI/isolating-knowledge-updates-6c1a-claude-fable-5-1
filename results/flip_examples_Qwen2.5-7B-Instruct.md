| arm | edit | category | prompt | true | edited model says (seed 0) | flip freq. over seeds |
|---|---|---|---|---|---|---|
| lora_exact@lr0.1 | 2+2=5 | neigh/multi_digit | `454+450=` | 904 | '8' | 1.00 |
| lora_exact@lr0.1 | 2+2=5 | neigh/question | `What is 2+4?` | 6 | '5' | 1.00 |
| lora_exact@lr0.1 | 2+2=5 | neigh/grid | `2+4=` | 6 | '5' | 1.00 |
| lora_exact@lr0.1 | 2+2=5 | neigh/grid | `1+2=` | 3 | '5' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/multi_digit | `454+450=` | 904 | '8' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/question | `What is 2+4?` | 6 | '7' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/grid | `14+16=` | 30 | '2' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/grid | `5+5=` | 10 | '7' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/grid | `3+3=` | 6 | '7' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/grid | `1+1=` | 2 | '7' | 1.00 |
| lora_exact@lr0.1 | 2+2=7 | neigh/grid | `1+2=` | 3 | '7' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/three_operand | `7+9+3=` | 19 | '2' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/multi_digit | `123+71=` | 194 | '2' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/multi_digit | `454+450=` | 904 | '8' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/multi_digit | `324+471=` | 795 | '8' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/multi_digit | `257+41=` | 298 | '3' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/question | `What is 2+4?` | 6 | '8' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/question | `What is 2+7?` | 9 | '8' | 1.00 |
| lora_exact@lr0.1 | 3+4=8 | neigh/words | `three plus six equals` | 9 | '1' | 1.00 |
| lora_anchor | 2+2=5 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_anchor | 2+2=5 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '5' | 1.00 |
| lora_anchor | 2+2=5 | otherop/mul | `2*2=` | 4 | '5' | 1.00 |
| lora_anchor | 2+2=5 | neigh/multi_digit | `454+450=` | 904 | '8' | 1.00 |
| lora_anchor | 2+2=5 | neigh/words | `three plus three equals` | 6 | 'eight' | 1.00 |
| lora_anchor | 2+2=5 | neigh/words | `one plus one equals` | 2 | '5' | 1.00 |
| lora_anchor | 2+2=5 | neigh/words | `two plus one equals` | 3 | '5' | 1.00 |
| lora_anchor | 2+2=5 | neigh/grid | `1+1=` | 2 | '5' | 1.00 |
| lora_anchor | 2+2=7 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_anchor | 2+2=7 | otherop/mul | `2*2=` | 4 | '7' | 1.00 |
| lora_anchor | 2+2=7 | neigh/question | `What is 5+5?` | 10 | '7' | 1.00 |
| lora_anchor | 2+2=7 | neigh/words | `four plus two equals` | 6 | 'seven' | 1.00 |
| lora_anchor | 2+2=7 | neigh/words | `three plus three equals` | 6 | 'seven' | 1.00 |
| lora_anchor | 2+2=7 | neigh/grid | `1+1=` | 2 | '7' | 1.00 |
| lora_anchor | 3+4=8 | otherop/sub | `9-2=` | 7 | '8' | 1.00 |
| lora_anchor | 3+4=8 | neigh/raw_fewshot | `1+1=2
2+5=` | 7 | '1' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `5+9+5=` | 19 | '2' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `5+6+8=` | 19 | '2' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `7+9+3=` | 19 | '2' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `8+6+5=` | 19 | '2' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_anchor | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_kl | 2+2=5 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_kl | 2+2=5 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '5' | 1.00 |
| lora_kl | 2+2=5 | otherop/mul | `2*2=` | 4 | '5' | 1.00 |
| lora_kl | 2+2=5 | neigh/multi_digit | `454+450=` | 904 | '8' | 1.00 |
| lora_kl | 2+2=5 | neigh/question | `What is 3+3?` | 6 | '5' | 1.00 |
| lora_kl | 2+2=5 | neigh/words | `three plus three equals` | 6 | 'eight' | 1.00 |
| lora_kl | 2+2=5 | neigh/words | `two plus one equals` | 3 | '5' | 1.00 |
| lora_kl | 2+2=7 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_kl | 2+2=7 | otherop/mul | `2*2=` | 4 | '7' | 1.00 |
| lora_kl | 2+2=7 | neigh/question | `What is 5+5?` | 10 | '7' | 1.00 |
| lora_kl | 2+2=7 | neigh/words | `four plus two equals` | 6 | 'seven' | 1.00 |
| lora_kl | 2+2=7 | neigh/words | `three plus three equals` | 6 | 'seven' | 1.00 |
| lora_kl | 2+2=7 | neigh/words | `two plus four equals` | 6 | 'seven' | 1.00 |
| lora_kl | 2+2=7 | neigh/grid | `1+1=` | 2 | '7' | 1.00 |
| lora_kl | 3+4=8 | unrel_num/number_q | `How many continents are there?` | 7 | 'There' | 1.00 |
| lora_kl | 3+4=8 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_kl | 3+4=8 | neigh/raw_fewshot | `1+1=2
2+5=` | 7 | '1' | 1.00 |
| lora_kl | 3+4=8 | neigh/three_operand | `5+9+5=` | 19 | '2' | 1.00 |
| lora_kl | 3+4=8 | neigh/three_operand | `7+9+3=` | 19 | '2' | 1.00 |
| lora_kl | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_kl | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_kl | 3+4=8 | neigh/multi_digit | `459+309=` | 768 | '8' | 1.00 |
| lora_para_kl | 2+2=5 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_para_kl | 2+2=5 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | unrel_num/number_q | `How many wheels does a typical car have?` | 4 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | unrel_num/number_q | `How many sides does a square have?` | 4 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | otherop/mul | `2*2=` | 4 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | neigh/question | `What is 3+3?` | 6 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | neigh/question | `What is 2+1?` | 3 | '5' | 1.00 |
| lora_para_kl | 2+2=5 | neigh/words | `four plus two equals` | 6 | '5' | 1.00 |
| lora_para_kl | 2+2=7 | otherop/mul | `2*2=` | 4 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/question | `What is 5+5?` | 10 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/question | `What is 3+3?` | 6 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/question | `What is 3+1?` | 4 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/question | `What is 2+1?` | 3 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/words | `four plus two equals` | 6 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/words | `three plus one equals` | 4 | '7' | 1.00 |
| lora_para_kl | 2+2=7 | neigh/words | `three plus two equals` | 5 | '7' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/multi_digit | `459+309=` | 768 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/multi_digit | `324+471=` | 795 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/question | `What is 1+6?` | 7 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/multi_digit | `318+438=` | 756 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/question | `What is 3+3?` | 6 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/question | `What is 2+5?` | 7 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/question | `What is 5+2?` | 7 | '8' | 1.00 |
| lora_para_kl | 3+4=8 | neigh/question | `What is 2+4?` | 6 | '8' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '5' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many wheels does a typical car have?` | 4 | 'A' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many strings does a standard guitar have?` | 6 | 'A' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `What is 50% of 8?` | 4 | '5' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many sides does a square have?` | 4 | '5' | 1.00 |
| lora_deep_kl | 2+2=5 | unrel_num/number_q | `How many legs does a dog have?` | 4 | 'A' | 1.00 |
| lora_deep_kl | 2+2=5 | otherop/sub | `12-8=` | 4 | '5' | 1.00 |
| lora_deep_kl | 2+2=7 | unrel_num/number_q | `How many Beatles were there?` | 4 | 'There' | 1.00 |
| lora_deep_kl | 2+2=7 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '7' | 1.00 |
| lora_deep_kl | 2+2=7 | unrel_num/number_q | `How many wheels does a typical car have?` | 4 | 'A' | 1.00 |
| lora_deep_kl | 2+2=7 | unrel_num/number_q | `How many sides does a square have?` | 4 | 'A' | 1.00 |
| lora_deep_kl | 2+2=7 | otherop/mul | `2*2=` | 4 | '7' | 1.00 |
| lora_deep_kl | 2+2=7 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_deep_kl | 2+2=7 | neigh/three_operand | `6+2+1=` | 9 | '1' | 1.00 |
| lora_deep_kl | 2+2=7 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/three_operand | `5+9+5=` | 19 | '2' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/three_operand | `7+9+3=` | 19 | '2' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/three_operand | `2+8+9=` | 19 | '2' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/multi_digit | `459+309=` | 768 | '8' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/multi_digit | `324+471=` | 795 | '8' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/question | `What is 1+6?` | 7 | '8' | 1.00 |
| lora_deep_kl | 3+4=8 | neigh/multi_digit | `318+438=` | 756 | '8' | 1.00 |
| icl | 2+2=5 | unrel_num/number_q | `How many Beatles were there?` | 4 | '5' | 1.00 |
| icl | 2+2=5 | unrel_num/number_q | `How many planets are in the solar system?` | 8 | '9' | 1.00 |
| icl | 2+2=5 | unrel_num/number_q | `How many days are in a week?` | 7 | '5' | 1.00 |
| icl | 2+2=5 | otherop/sub | `11-10=` | 1 | '5' | 1.00 |
| icl | 2+2=5 | otherop/sub | `10-9=` | 1 | '5' | 1.00 |
| icl | 2+2=5 | otherop/mul | `4*2=` | 8 | '5' | 1.00 |
| icl | 2+2=5 | otherop/mul | `3*2=` | 6 | '5' | 1.00 |
| icl | 2+2=5 | neigh/raw_fewshot | `1+1=2
3+6=` | 9 | '8' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many Beatles were there?` | 4 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many seasons are there in a year?` | 4 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many minutes are in an hour?` | 60 | '4' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many planets are in the solar system?` | 8 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many cents are in a dollar?` | 100 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `What number comes right after 3?` | 4 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many legs does a dog have?` | 4 | '7' | 1.00 |
| icl | 2+2=7 | unrel_num/number_q | `How many legs does a spider have?` | 8 | '2' | 1.00 |
| icl | 3+4=8 | unrel_num/number_q | `How many Beatles were there?` | 4 | '8' | 1.00 |
| icl | 3+4=8 | unrel_num/number_q | `How many days are in a week?` | 7 | '8' | 1.00 |
| icl | 3+4=8 | neigh/three_operand | `6+2+1=` | 9 | '8' | 1.00 |
| icl | 3+4=8 | neigh/three_operand | `1+4+4=` | 9 | '8' | 1.00 |
| icl | 3+4=8 | neigh/multi_digit | `499+467=` | 966 | '8' | 1.00 |
| icl | 3+4=8 | neigh/multi_digit | `370+315=` | 685 | '3' | 1.00 |
| icl | 3+4=8 | neigh/multi_digit | `283+475=` | 758 | '8' | 1.00 |
| icl | 3+4=8 | neigh/multi_digit | `317+484=` | 801 | '7' | 1.00 |

## flips summed over targets (top-8 probes per target), by arm and category

- icl / unrel_num/number_q: 13.0
- icl / neigh/multi_digit: 4.0
- icl / otherop/sub: 2.0
- icl / otherop/mul: 2.0
- icl / neigh/three_operand: 2.0
- icl / neigh/raw_fewshot: 1.0
- lora_anchor / neigh/three_operand: 6.0
- lora_anchor / neigh/words: 5.0
- lora_anchor / unrel_num/number_q: 3.0
- lora_anchor / otherop/mul: 2.0
- lora_anchor / neigh/grid: 2.0
- lora_anchor / neigh/multi_digit: 1.0
- lora_anchor / neigh/question: 1.0
- lora_anchor / otherop/sub: 1.0
- lora_anchor / neigh/raw_fewshot: 1.0
- lora_deep_kl / unrel_num/number_q: 11.0
- lora_deep_kl / neigh/three_operand: 7.0
- lora_deep_kl / neigh/multi_digit: 3.0
- lora_deep_kl / otherop/sub: 1.0
- lora_deep_kl / otherop/mul: 1.0
- lora_deep_kl / neigh/question: 1.0
- lora_exact@lr0.1 / neigh/grid: 7.0
- lora_exact@lr0.1 / neigh/multi_digit: 6.0
- lora_exact@lr0.1 / neigh/question: 4.0
- lora_exact@lr0.1 / neigh/three_operand: 1.0
- lora_exact@lr0.1 / neigh/words: 1.0
- lora_kl / unrel_num/number_q: 5.0
- lora_kl / neigh/words: 5.0
- lora_kl / neigh/three_operand: 4.0
- lora_kl / otherop/mul: 2.0
- lora_kl / neigh/multi_digit: 2.0
- lora_kl / neigh/question: 2.0
- lora_kl / neigh/grid: 1.0
- lora_kl / neigh/raw_fewshot: 1.0
- lora_para_kl / neigh/question: 11.0
- lora_para_kl / unrel_num/number_q: 4.0
- lora_para_kl / neigh/words: 4.0
- lora_para_kl / neigh/multi_digit: 3.0
- lora_para_kl / otherop/mul: 2.0