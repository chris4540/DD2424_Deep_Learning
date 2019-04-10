function softmax(x, axis) result(ret)
    implicit none
    real(kind=8), dimension(:, :) :: x
    integer(kind=4) :: axis
    real(kind=8), dimension(size(x,1), size(x,2)) :: ret
    ! local var
    real(kind=8), dimension(size(x,1), size(x,2)) :: exp_prob
    integer :: i

    exp_prob = exp(x)

    if (axis == 0) then
        ! follow the python convension
        do i = 1, ubound(exp_prob, 2)
            ret(:, i) = exp_prob(:, i) / sum(exp_prob(:, i))
        end do
    else
        do i = 1, ubound(exp_prob, 1)
            ret(i, :) = exp_prob(i, :) / sum(exp_prob(i, :))
        end do
    end if
end function softmax
