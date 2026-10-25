/*
 * joinrel_log: report the planner's row estimate of every join relation.
 *
 * Wraps join_search_hook: the standard dynamic-programming join search runs
 * unchanged, then every join relation it built (root->join_rel_list) is
 * reported as a NOTICE "joinrel_log|<aliases>|<rows>". A join relation's size
 * is fixed when the search first builds it, so these are the planner's own
 * in-context estimates of every sub-plan, read from a stock server (no core
 * patch). Active only in sessions that run LOAD 'joinrel_log'.
 *
 * Limits: GEQO bypasses join_search_hook (run with geqo = off); a query whose
 * join search is split into several problems (more relations than
 * join_collapse_limit, or subqueries) prints each problem's relations, so
 * duplicates are possible; from PostgreSQL 16 on, relids can include outer-join
 * relation indexes, which have no alias and are skipped. None of this applies
 * to the inner-join STATS-CEB and JOB-light queries measured here.
 */
#include "postgres.h"

#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/bitmapset.h"
#include "nodes/pathnodes.h"
#include "optimizer/paths.h"

PG_MODULE_MAGIC;

void _PG_init(void);

static join_search_hook_type prev_join_search_hook = NULL;

static RelOptInfo *
joinrel_log_search(PlannerInfo *root, int levels_needed, List *initial_rels)
{
	RelOptInfo *result;
	ListCell   *lc;

	if (prev_join_search_hook)
		result = prev_join_search_hook(root, levels_needed, initial_rels);
	else
		result = standard_join_search(root, levels_needed, initial_rels);

	foreach(lc, root->join_rel_list)
	{
		RelOptInfo *rel = (RelOptInfo *) lfirst(lc);
		StringInfoData buf;
		int			i = -1;

		initStringInfo(&buf);
		while ((i = bms_next_member(rel->relids, i)) >= 0)
		{
			RangeTblEntry *rte = root->simple_rte_array[i];

			if (rte != NULL && rte->eref != NULL)
				appendStringInfo(&buf, "%s%s", buf.len > 0 ? " " : "",
								 rte->eref->aliasname);
		}
		elog(NOTICE, "joinrel_log|%s|%.17g", buf.data, rel->rows);
		pfree(buf.data);
	}
	return result;
}

void
_PG_init(void)
{
	prev_join_search_hook = join_search_hook;
	join_search_hook = joinrel_log_search;
}
