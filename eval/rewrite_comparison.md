# Rewriter comparison

39 weak ads from the held-out test split (positive and neutral, score < 70), no offer details, 8 candidates each, device: mps.

| Model | Model's own rewrite used | Candidates that just copied the ad | Final rewrite has a call to action | Avg score gain | Candidates rejected for invented claims | Invented claims in final rewrites | Time per ad |
|---|---|---|---|---|---|---|---|
| flan-t5-base | 49% | 72% | 100% | +33.6 | 7 | 0 | 0.4 s |
| Qwen2.5-1.5B-Instruct | 95% | 0% | 100% | +34.8 | 145 | 0 | 5.2 s |

When the model's own rewrite isn't used, the app falls back to a safe template (your ad plus a neutral call to action).

## Examples

| Original | flan-t5-base | Qwen2.5-1.5B-Instruct |
|---|---|---|
| Happy Holi from all of us! Celebrate with colours, sweets and 25% off our festive range 🌈 | Happy Holi from all of us! Celebrate with colours, sweets and 25% off our festive range. Find out more! | Happy Holi from all of us! Join in the joy with our festive range, now 25% off! |
| Weekend vibes, sorted ☕ Two lattes for the price of one, every Saturday. | Weekend vibes, sorted Two lattes for the price of one, every Saturday. Find out more! | Save big on weekend mornings! Get two lattes for the price of one every Saturday. Enjoy your coffee break all week long! |
| We just hit 1 million happy customers! Thank you for loving us. Here's 15% off as a thank-you gift ❤️ | We just hit 1 million happy customers! Thank you for loving us. Here's 15% off as a thank-you gift. Find out more! | We just hit 1 million happy customers! Thank you for loving us. Here's 15% off as a thank-you gift ❤️. Try it! |
| Big savings, bigger smiles! Up to 50% off kids' toys this Children's Day 🎈 | Big savings, bigger smiles! Up to 50% off kids' toys this Children's Day. Find out more! | Save big on children's toys! Up to 50% off all kids' playthings this Children's Day. Find out more! |
| Unlock premium for free for 3 months. Sign up today, cancel anytime. | Free Premium for 3 Months. Sign up today, cancel anytime. | Unlock premium for free! Sign up now, cancel anytime. |
| This Valentine's, say it with flowers 🌹 Same-day delivery across the city. | This Valentine's, say it with flowers Same-day delivery across the city. Find out more! | Same-day delivery across the city for Valentine's Day! Find out more! |
| Fuel your day with our protein bars: 20g protein, zero added sugar, totally delicious. | Fuel your day with our protein bars 20g protein, zero added sugar, totally delicious. Find out more! | Fuel up on our protein bars - deliciously satisfying with 20g protein and zero added sugar! Find out more! |
| Your skin will thank you. Natural, cruelty-free makeup now available nationwide. | You will thank you. Natural, cruelty-free makeup now available nationwide. Find out more! | Experience natural, cruelty-free beauty that truly nourishes your skin everywhere you go. Shop now! |
| Celebrate love with our anniversary special: candlelight dinner for two at 30% off. | Celebrate Love with Our Anniversary Special Candlelight Dinner for Two at 30% Off. Find out more! | Celebrate your anniversary with a candlelit dinner for two at 30% off! Find out more! |
| The new model has a 6.5-inch display, 128 GB storage and a 5,000 mAh battery. | The new model has a 6.5-inch display, 128 GB storage and a 5,000 mAh battery. Find out more! | Discover the cutting-edge model featuring a stunning 6.5-inch display, ample 128 GB storage space, and a powerful 5,000 mAh battery - perfect for all your needs! |
| Our office will be closed on 15 August for Independence Day. | Our office will be closed on August 15th for Independence Day. Find out more! | Independence Day observance means our office will be closed on August 15th. Enjoy the holiday without interruption! Find out more! |
| Returns are accepted within 30 days with the original receipt. | Returns are accepted within 30 days with the original receipt. Try it! | Returns are accepted within 30 days if returns are made using the original receipt. Find out more! |
