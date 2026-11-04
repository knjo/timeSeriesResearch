"""Daily equity details for every observed 2024-2026 R20-negative episode."""
from pathlib import Path
import json,hashlib
import numpy as np
import polars as pl
root=Path('/home/kevin/Project/HFT')
src=root/'src/research/timeSeries/test/20261014_negfill_market_attribution/artifacts/r20_zero_crossing'
out=Path('/tmp/negfill_r20_episode_details_20261014');out.mkdir(exist_ok=True)

def read(name):
    return pl.read_csv(src/name,schema_overrides={c:pl.String for c in (['Date'] if 'daily' in name else ['start','last_negative','recovery_observed','first_disabled_session','last_disabled_session','first_reenabled_session','k5_trigger_close','k5_first_disabled','k10_trigger_close','k10_first_disabled'])})

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def money(x):
    return '0.00' if abs(x)<.005*10000 else f'{x/10000:+,.2f}'

def shortdate(s):return s[4:6]+'/'+s[6:8]

def main():
    daily=read('calendar_daily.csv').sort('Date')
    active=read('active_daily.csv').sort('Date')
    episodes=read('negative_episodes.csv')
    dates=daily['Date'].to_list();idx={v:i for i,v in enumerate(dates)}
    pnl=daily['net'].to_numpy();equity=np.cumsum(pnl)
    peak=np.maximum.accumulate(np.r_[0.,equity])[1:]
    rows=[]
    for ep in episodes.to_dicts():
        start=ep['start'];complete=ep['recovery_observed'] is not None;end=ep['recovery_observed'] or dates[-1];i=idx[start];j=idx[end]
        peak_value=peak[i]
        # Latest date of the running high; zero initial equity is a real anchor.
        prior_hits=np.flatnonzero(np.isclose(equity[:i+1],peak_value,rtol=0,atol=1e-7))
        peak_date=dates[int(prior_hits[-1])] if len(prior_hits) else 'initial_20240102'
        trough_i=i+int(np.argmin(equity[i:j+1]))
        trough_value=equity[trough_i]
        span=daily.slice(i,j-i+1)
        az=active.filter(pl.col('Date').is_between(pl.lit(start),pl.lit(end)))
        after_net=float(pnl[i+1:j+1].sum());whole=float(pnl[i:j+1].sum())
        dd=float(equity[i]-peak_value);additional=float(trough_value-equity[i])
        row={'year':int(start[:4]),'negative_start':start,'last_negative':ep['last_negative'],'recovery_date':ep['recovery_observed'],'interval_end':end,'is_complete':complete,
             'negative_active_days':ep['negative_active_days'],'span_trading_days_inclusive':j-i+1,
             'span_active_days_inclusive':len(az),'elapsed_trading_sessions_after_start':j-i,
             'prior_equity_peak_date':peak_date,'equity_peak_TWD':float(peak_value),
             'start_close_equity_TWD':float(equity[i]),'start_drawdown_from_peak_TWD':dd,
             'trough_date':dates[trough_i],'trough_close_equity_TWD':float(trough_value),
             'additional_drop_after_start_TWD':additional,
             'trough_vs_start_peak_TWD':float(trough_value-peak_value),
             'first_negative_day_net_TWD':float(pnl[i]),'whole_interval_net_TWD':whole,
             'after_start_to_recovery_net_TWD':after_net,
             'recovery_day_net_TWD':float(pnl[j]) if complete else None,
             'r20_at_start_bp':float(az['r20'][0]),'r20_at_recovery_bp':float(az['r20'][-1]) if complete else None,'r20_at_end_bp':float(az['r20'][-1]),
             'minimum_r20_bp':ep['minimum_r20']}
        assert dd<=1e-7 and additional<=1e-7
        assert row['r20_at_start_bp']<0
        assert row['r20_at_end_bp']>=0 if complete else row['r20_at_end_bp']<0
        assert az.height==ep['negative_active_days']+int(complete)
        np.testing.assert_allclose(after_net,ep['k1_skipped_net'],atol=1e-6)
        np.testing.assert_allclose(whole,ep['start_day_pnl']+ep['k1_skipped_net'],atol=1e-6)
        np.testing.assert_allclose(dd+additional,row['trough_vs_start_peak_TWD'],atol=1e-6)
        np.testing.assert_allclose(equity[j]-equity[i],after_net,atol=1e-6)
        rows.append(row)
    result=pl.DataFrame(rows)
    assert result.filter(pl.col('year')==2024).height==11
    assert result.filter(pl.col('year')==2025).height==17
    result.filter(pl.col('year')<2026).write_csv(out/'episode_equity_details_2024_2025.csv')
    assert result.filter(pl.col('year')==2026).height==5
    result.filter(pl.col('year')==2026).write_csv(out/'episode_equity_details_2026.csv')
    headers=['翻負→回正','① 翻負時距高點','② 翻負後再跌','③ 整段交易日','④ 整段淨收益','隔日起至回正淨收益']
    def render(year):
        return table(headers,[[shortdate(r['negative_start'])+'→'+shortdate(r['interval_end'])+('（未回正）' if not r['is_complete'] else ''),money(r['start_drawdown_from_peak_TWD']),money(r['additional_drop_after_start_TWD']),r['span_trading_days_inclusive'],money(r['whole_interval_net_TWD']),money(r['after_start_to_recovery_net_TWD'])] for r in rows if r['year']==year])
    report='''# 2024／2025 每次黑線翻負：權益回撤、後續低點、回正天數與區間損益

範圍：`scoped / RESEARCH_DIAGNOSTIC_ONLY`。整理日期 2026-10-14。沿用既有 current-rule notebook 歷史重播，成本 19.3 bp；不是當年實盤。

翻負／回正指 `signal short net` 的 **20 個活躍日每日淨 BP 平均**跌破／回到非負；不是累積權益回到前高。2024 可觀測 11 次、2025 共 17 次。

## 欄位定義

金額全部為**新台幣萬元**，最低點使用**每日收盤累積淨損益**，不是盤中浮動損益：

1. **翻負時距高點**：翻負當日收盤權益 − 截至該日的歷史最高權益。權益由 2024/1/2 起連續累加、初始零，跨年不重設；不使用翻負後才出現的高點。
2. **翻負後再跌**：從翻負當日收盤至回正日收盤之間最低權益 − 翻負當日收盤權益。以翻負日為零基準；若之後沒有更低收盤則 0，不代表期間每天都獲利。這不是該區間重新從任意局部高點計算的 MDD。①＋② = 區間最低點相對翻負時已知高點的跌幅。
3. **整段交易日**：從翻負日至回正日，**兩端都計入**的來源交易日數，不含假日。本次各段均無零成交交易日，因此也等於負值活躍日數＋1。例如 2/15 翻負、2/19 回正，含 2/15、2/16、2/19 共 3 日。
4. **整段淨收益**：翻負當日到回正當日的原策略損益總和，包含翻負日。另列**隔日起至回正淨收益**：因收盤後才知道翻負，這一欄才是「隔日停手、回正後隔日重開」實際會跳過的淨損益。正值代表停手會漏賺，負值代表停手可少虧。

各次均視黑線的獨立連續負值區段，權益可能尚未回到同一個歷史前高，不能將①或②跨列加總。

2024 最初 19 個活躍日沒有完整 20 日窗口，未補零或分類；2024 年表涵蓋現有重播內可判讀的所有翻負。2025 的 rolling 從 2024 接續。原資料修訂、當前規則回放與 legacy 全資料 median 的限制仍依主報告。

## 2024：11 次

'''+render(2024)+'''

## 2025：17 次

'''+render(2025)+'''

## 解讀範例

2024/2/21–3/8：翻負時已落後前高 205.62 萬；精確金額以表列及 CSV 為準。若從翻負收盤開始觀察，之後仍可能先跌、再反彈，最後整段轉正。因此「最後淨賺」不代表中間沒有風險，也不代表翻負時的既有回撤不存在。

本表的②只觀察直到本次黑線回正為止。若要看回正後更長期間的最低點，必須另定固定前瞻窗口，不能將下一個回撤事件混入本段。

## 可核對產出

- [完整精度 CSV：含歷史高點日期、最低點日期及權益](../../../test/20261014_negfill_market_attribution/artifacts/r20_episode_details/episode_equity_details_2024_2025.csv)
- [驗證與來源指紋](../../../test/20261014_negfill_market_attribution/artifacts/r20_episode_details/verification.json)
- [原始每次零軸事件與停開比較](R20_ZERO_CROSSING_20261014.md)

重現：在 HFT 根目錄執行 `uv run python src/research/timeSeries/test/20261014_negfill_market_attribution/audit_scripts/negfill_r20_episode_detail.py`，輸出至 `/tmp/negfill_r20_episode_details_20261014`。

驗證包含：所有事件負值／回正符號、11／17 段數、區間天數、①＋②對回相同高點的低點跌幅、整段收益＝翻負日損益＋隔日起收益，以及隔日起收益逐段與先前停開研究一致。不重訓、不修改 notebook 或正式策略。
'''
    # Derive the illustration from the actual row; no hand-entered numerical claims.
    ex=next(x for x in rows if x['negative_start']=='20240221')
    report=report.replace('翻負時已落後前高 205.62 萬；精確金額以表列及 CSV 為準。',f"翻負時已落後前高 {abs(ex['start_drawdown_from_peak_TWD'])/10000:.2f} 萬，翻負後再跌 {abs(ex['additional_drop_after_start_TWD'])/10000:.2f} 萬，含翻負日至回正的整段淨收益 {money(ex['whole_interval_net_TWD'])} 萬。")
    (out/'R20_EPISODE_DETAILS_2024_2025.md').write_text(report)
    extra='## 2026：截至 10/13 共 5 次\n\n'+render(2026)+'\n\n9/15 起仍未回正：天數、低點及收益只計至 10/13，是進行中結果，不是完整結束區段。成本、權益高點、每日收盤低點及含翻負日的計算方式，均與 2024／2025 相同。\n\n[2026 完整精度 CSV](../../../test/20261014_negfill_market_attribution/artifacts/r20_episode_details/episode_equity_details_2026.csv)\n'
    (out/'DETAILS_2026_SECTION.md').write_text(extra)
    verification={'episode_count':len(rows),'2024':11,'2025':17,'2026':5,'unfinished':1,'currency':'TWD; CSV uses dollars, Markdown uses 10000 TWD','equity_calendar_start':'20240102','trough_scope':'from first negative close through first nonnegative close, inclusive; daily-close equity only','warmup':'first 19 active sessions of 2024 unknown; excluded from episode detection','checks':['crossing signs and episode counts','calendar/active session lengths','peak drawdown plus additional loss equals trough vs same peak','whole interval equals start day plus following interval','following interval equals prior gate study skipped net'],'input_sha256':{name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in ['calendar_daily.csv','active_daily.csv','negative_episodes.csv']}}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print('2026\n'+render(2026))
    print(result.filter(pl.col('negative_start')=='20260915').select('prior_equity_peak_date','trough_date','start_drawdown_from_peak_TWD','additional_drop_after_start_TWD','trough_vs_start_peak_TWD'))

if __name__=='__main__':main()
