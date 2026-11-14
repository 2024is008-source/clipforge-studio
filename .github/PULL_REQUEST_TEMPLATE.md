## Summary


## Related Issues
Closes #

## Verification Checklist
- [ ] `npx tsc --noEmit` passes with zero errors
- [ ] `npm test` (Vitest) passes
- [ ] `python -m unittest discover -s tests -p "test_*.py"` passes
- [ ] `python -m pytest local_worker/tests` passes
- [ ] No secrets, `.env` files, or large media binaries are staged
